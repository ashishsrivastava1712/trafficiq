import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib, json
from models.mstn_model import MSTN

SEQ_LEN    = 24
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Feature Engineering
df = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')
df['date_time']  = pd.to_datetime(df['date_time'])
df['hour']       = df['date_time'].dt.hour
df['dow']        = df['date_time'].dt.dayofweek
df['month']      = df['date_time'].dt.month
df['is_weekend'] = (df['dow'] >= 5).astype(float)
df['is_holiday'] = (df['holiday'] != 'None').astype(float)
df['temp_c']     = df['temp'] - 273.15
df['rush_hour']  = df['hour'].apply(lambda h: 1.0 if h in [7,8,9,16,17,18] else 0.0)

le = LabelEncoder()
df['weather_enc'] = le.fit_transform(df['weather_main']).astype(float)

FEATURES = ['hour','dow','month','is_weekend','is_holiday',
            'temp_c','rain_1h','snow_1h','clouds_all','weather_enc','rush_hour']
TARGET = 'traffic_volume'

df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(df[FEATURES].values)
y_scaled = scaler_y.fit_transform(df[[TARGET]].values).flatten()

class TrafficSeqDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X, self.y, self.sl = X, y, seq_len
    def __len__(self):
        return len(self.X) - self.sl
    def __getitem__(self, i):
        return (torch.FloatTensor(self.X[i:i+self.sl]),
                torch.FloatTensor([self.y[i+self.sl]]))

n = len(X_scaled)
split_train = int(n * 0.7)
split_val   = int(n * 0.85)

train_ds = TrafficSeqDataset(X_scaled[:split_train],        y_scaled[:split_train],        SEQ_LEN)
val_ds   = TrafficSeqDataset(X_scaled[split_train:split_val], y_scaled[split_train:split_val], SEQ_LEN)
test_ds  = TrafficSeqDataset(X_scaled[split_val:],           y_scaled[split_val:],           SEQ_LEN)

train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(val_ds,   BATCH_SIZE)
test_dl  = DataLoader(test_ds,  BATCH_SIZE)

model = MSTN(input_dim=len(FEATURES), seq_len=SEQ_LEN, dropout=0.3).to(DEVICE)
print(f"MSTN Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
criterion = nn.HuberLoss()

best_val = float('inf')
for epoch in range(EPOCHS):
    model.train()
    t_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        t_loss += loss.item()

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            v_loss += criterion(model(xb), yb.squeeze()).item()

    t_loss /= len(train_dl)
    v_loss /= len(val_dl)
    scheduler.step(v_loss)

    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), 'models/mstn_best.pth')

    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {t_loss:.5f} | Val: {v_loss:.5f}")

# Evaluate
model.load_state_dict(torch.load('models/mstn_best.pth'))
model.eval()
preds, trues = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds.extend(model(xb.to(DEVICE)).cpu().numpy())
        trues.extend(yb.numpy())

preds = scaler_y.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
trues = scaler_y.inverse_transform(np.array(trues).reshape(-1,1)).flatten()

mae  = mean_absolute_error(trues, preds)
rmse = np.sqrt(mean_squared_error(trues, preds))
r2   = r2_score(trues, preds)

print("\n" + "="*50)
print("MSTN TEST RESULTS")
print("="*50)
print(f"  MAE  : {mae:.2f} vehicles/hour")
print(f"  RMSE : {rmse:.2f} vehicles/hour")
print(f"  R2   : {r2:.4f}  ({r2*100:.2f}%)")
print("="*50)

joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
joblib.dump(le,       'models/label_encoder.pkl')

config = {
    'seq_len': SEQ_LEN,
    'input_dim': len(FEATURES),
    'features': FEATURES,
    'mae': mae, 'rmse': rmse, 'r2': r2
}
json.dump(config, open('models/config.json', 'w'))
print("Models saved to models/")