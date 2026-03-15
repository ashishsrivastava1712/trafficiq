import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── Load & Feature Engineer ──────────────────────────────────
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

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

results = {}

# ── Random Forest ────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
p = rf.predict(X_test)
results['Random Forest'] = (mean_absolute_error(y_test,p), np.sqrt(mean_squared_error(y_test,p)), r2_score(y_test,p))

# ── XGBoost ──────────────────────────────────────────────────
print("Training XGBoost...")
xgb_m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
xgb_m.fit(X_train, y_train)
p = xgb_m.predict(X_test)
results['XGBoost'] = (mean_absolute_error(y_test,p), np.sqrt(mean_squared_error(y_test,p)), r2_score(y_test,p))

# ── LightGBM ─────────────────────────────────────────────────
print("Training LightGBM...")
lgb_m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
lgb_m.fit(X_train, y_train)
p = lgb_m.predict(X_test)
results['LightGBM'] = (mean_absolute_error(y_test,p), np.sqrt(mean_squared_error(y_test,p)), r2_score(y_test,p))

# ── LSTM ─────────────────────────────────────────────────────
print("Training LSTM...")
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEQ = 24
def make_seq(X, y, s):
    Xs, ys = [], []
    for i in range(len(X)-s):
        Xs.append(X[i:i+s]); ys.append(y[i+s])
    return np.array(Xs,dtype=np.float32), np.array(ys,dtype=np.float32)

Xtr_s2, ytr_s2 = make_seq(X_train_s, y_train, SEQ)
Xte_s2, yte_s2 = make_seq(X_test_s,  y_test,  SEQ)

class LSTMModel(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.lstm = nn.LSTM(inp, 64, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
lstm = LSTMModel(X_train_s.shape[1]).to(dev)
opt  = torch.optim.Adam(lstm.parameters(), lr=1e-3)
dl   = DataLoader(TensorDataset(torch.FloatTensor(Xtr_s2), torch.FloatTensor(ytr_s2)), 64, shuffle=True)
for ep in range(10):
    lstm.train()
    for xb,yb in dl:
        opt.zero_grad()
        nn.HuberLoss()(lstm(xb.to(dev)), yb.to(dev)).backward()
        opt.step()
lstm.eval()
with torch.no_grad():
    p = lstm(torch.FloatTensor(Xte_s2).to(dev)).cpu().numpy()
results['LSTM'] = (mean_absolute_error(yte_s2,p), np.sqrt(mean_squared_error(yte_s2,p)), r2_score(yte_s2,p))

# ── MSTN (YOUR MODEL) ─────────────────────────────────────────
print("Training MSTN (your novel model)...")
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from models.mstn_model import MSTN

y_mean, y_std = y_train.mean(), y_train.std()
ytr_n = (y_train - y_mean) / y_std
yte_n = (y_test  - y_mean) / y_std

Xtr_m, ytr_m = make_seq(X_train_s, ytr_n, SEQ)
Xte_m, yte_m = make_seq(X_test_s,  yte_n, SEQ)

mstn = MSTN(input_dim=X_train_s.shape[1], seq_len=SEQ).to(dev)
opt2 = torch.optim.AdamW(mstn.parameters(), lr=1e-3, weight_decay=1e-4)
sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, patience=3, factor=0.5)
dl2  = DataLoader(TensorDataset(torch.FloatTensor(Xtr_m), torch.FloatTensor(ytr_m)), 64, shuffle=True)

best, best_ep = float('inf'), 0
for ep in range(20):
    mstn.train()
    ep_loss = 0
    for xb,yb in dl2:
        opt2.zero_grad()
        loss = nn.HuberLoss()(mstn(xb.to(dev)), yb.to(dev))
        loss.backward()
        nn.utils.clip_grad_norm_(mstn.parameters(), 1.0)
        opt2.step()
        ep_loss += loss.item()
    ep_loss /= len(dl2)
    sch.step(ep_loss)
    if ep_loss < best:
        best = ep_loss
        torch.save(mstn.state_dict(), 'models/mstn_bench_best.pth')
    print(f"  MSTN Epoch {ep+1:2d}/20 | Loss: {ep_loss:.5f}")

mstn.load_state_dict(torch.load('models/mstn_bench_best.pth'))
mstn.eval()
with torch.no_grad():
    p_n = mstn(torch.FloatTensor(Xte_m).to(dev)).cpu().numpy()
p = p_n * y_std + y_mean
results['MSTN-BiLSTM (Yours)'] = (mean_absolute_error(yte_m*y_std+y_mean, p),
                                   np.sqrt(mean_squared_error(yte_m*y_std+y_mean, p)),
                                   r2_score(yte_m*y_std+y_mean, p))

# ── PRINT RESULTS ─────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Acc%':>8}")
print("="*65)
for name, (mae, rmse, r2) in results.items():
    marker = " ◄ YOUR MODEL" if "MSTN" in name else ""
    print(f"{name:<25} {mae:>8.1f} {rmse:>8.1f} {r2:>8.4f} {max(0,r2)*100:>7.2f}%{marker}")
print("="*65)