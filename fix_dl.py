# =============================================================
# FIXED DL BENCHMARK
# Same 80/20 split, proper scaling fix
# =============================================================
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.mstn_model import MSTN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ    = 24
print(f"Device: {DEVICE}")

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')
df['date_time']  = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)

df['hour']       = df['date_time'].dt.hour
df['dow']        = df['date_time'].dt.dayofweek
df['month']      = df['date_time'].dt.month
df['quarter']    = df['date_time'].dt.quarter
df['is_weekend'] = (df['dow'] >= 5).astype(float)
df['is_holiday'] = (df['holiday'] != 'None').astype(float)
df['temp_c']     = df['temp'] - 273.15
df['rush_hour']  = df['hour'].apply(lambda h: 1.0 if h in [7,8,9,16,17,18] else 0.0)
df['night']      = df['hour'].apply(lambda h: 1.0 if h in range(0,6) else 0.0)
df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin']    = np.sin(2 * np.pi * df['dow'] / 7)
df['dow_cos']    = np.cos(2 * np.pi * df['dow'] / 7)
df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)

le = LabelEncoder()
df['weather_enc'] = le.fit_transform(df['weather_main']).astype(float)
df = df.dropna().reset_index(drop=True)

FEATURES = ['hour','dow','month','quarter','is_weekend','is_holiday',
            'temp_c','rain_1h','snow_1h','clouds_all','weather_enc',
            'rush_hour','night','hour_sin','hour_cos',
            'dow_sin','dow_cos','month_sin','month_cos']

X = df[FEATURES].values.astype(np.float32)
y = df['traffic_volume'].values.astype(np.float32)

# ── 2. 80/20 SPLIT ────────────────────────────────────────────
n         = len(X)
train_end = int(n * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_test,  y_test  = X[train_end:], y[train_end:]

print(f"\nTotal    : {n:,}")
print(f"Train 80%: {len(X_train):,} samples")
print(f"Test  20%: {len(X_test):,}  samples")
print(f"Train period: {df['date_time'].iloc[0]} → {df['date_time'].iloc[train_end-1]}")
print(f"Test  period: {df['date_time'].iloc[train_end]} → {df['date_time'].iloc[-1]}")

# ── 3. SCALE X ONLY — y raw rakho! ───────────────────────────
# KEY FIX: y ko normalize mat karo
# Direct traffic_volume use karo
scaler_X = StandardScaler()
X_tr_s   = scaler_X.fit_transform(X_train).astype(np.float32)
X_te_s   = scaler_X.transform(X_test).astype(np.float32)

# Scale y separately — sirf 0-1 range mein
y_min = float(y_train.min())
y_max = float(y_train.max())

def scale_y(y):   return (y - y_min) / (y_max - y_min)
def unscale_y(y): return y * (y_max - y_min) + y_min

y_tr_s = scale_y(y_train)
y_te_s = scale_y(y_test)

# ── 4. MAKE SEQUENCES ─────────────────────────────────────────
def make_seq(X, y, s):
    Xs, ys = [], []
    for i in range(len(X) - s):
        Xs.append(X[i:i+s])
        ys.append(y[i+s])
    return np.array(Xs, np.float32), np.array(ys, np.float32)

Xtr_s, ytr_s = make_seq(X_tr_s, y_tr_s, SEQ)
Xte_s, yte_s = make_seq(X_te_s, y_te_s, SEQ)

# Actual traffic_volume for test (unscaled) — REAL values!
y_test_actual = y_test[SEQ:]

print(f"\nTrain sequences: {len(Xtr_s):,}")
print(f"Test  sequences: {len(Xte_s):,}")
print(f"y_test range   : {y_test_actual.min():.0f} → {y_test_actual.max():.0f} vehicles/hour")

tr_dl = DataLoader(
    TensorDataset(torch.FloatTensor(Xtr_s),
                  torch.FloatTensor(ytr_s)),
    batch_size=64, shuffle=True)

INP = X_tr_s.shape[1]
results = {}

def evaluate(name, actual, predicted):
    """Compare actual traffic_volume vs predicted traffic_volume"""
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100

    def level(v):
        if v < 1000: return 'LOW'
        if v < 3000: return 'MEDIUM'
        if v < 5000: return 'HIGH'
        return 'SEVERE'

    al   = [level(v) for v in actual]
    pl   = [level(v) for v in predicted]
    cacc = np.mean([a==b for a,b in zip(al,pl)]) * 100

    results[name] = {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'MAPE': mape, 'ClassAcc': cacc,
        'preds': predicted
    }
    marker = "  ◄ YOUR MODEL" if "MSTN" in name else ""
    print(f"  ✓ {name:<28} "
          f"MAE={mae:7.1f}  "
          f"RMSE={rmse:7.1f}  "
          f"R²={r2:.4f} ({max(0,r2)*100:.2f}%)  "
          f"ClassAcc={cacc:.1f}%"
          f"{marker}")

def train_and_predict(model, epochs=15, lr=1e-3):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss()
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

    # Predict → unscale → actual traffic_volume
    model.eval()
    with torch.no_grad():
        p_scaled = model(torch.FloatTensor(Xte_s).to(DEVICE)).cpu().numpy()
    return unscale_y(p_scaled)  # back to real traffic_volume!

# ── 5. LSTM ───────────────────────────────────────────────────
print("\n── DL MODELS (Fixed) ──────────────────────────────────────")

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INP, 128, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid())  # Sigmoid → 0 to 1
    def forward(self, x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training LSTM...")
m = LSTMModel().to(DEVICE)
train_and_predict(m)  # warmup
p = train_and_predict(m)
evaluate("LSTM", y_test_actual, p)

# ── 6. BiLSTM ─────────────────────────────────────────────────
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INP, 64, 2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.fc   = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training BiLSTM...")
m = BiLSTMModel().to(DEVICE)
train_and_predict(m)
p = train_and_predict(m)
evaluate("BiLSTM", y_test_actual, p)

# ── 7. GRU ────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INP, 128, 2, batch_first=True, dropout=0.2)
        self.fc  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        o,_ = self.gru(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training GRU...")
m = GRUModel().to(DEVICE)
train_and_predict(m)
p = train_and_predict(m)
evaluate("GRU", y_test_actual, p)

# ── 8. CNN-LSTM ───────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = nn.Sequential(
            nn.Conv1d(INP, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),  nn.ReLU())
        self.lstm = nn.LSTM(64, 64, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training CNN-LSTM...")
m = CNNLSTMModel().to(DEVICE)
train_and_predict(m)
p = train_and_predict(m)
evaluate("CNN-LSTM", y_test_actual, p)

# ── 9. Transformer ────────────────────────────────────────────
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(INP, 128)
        enc = nn.TransformerEncoderLayer(
            128, 4, 256, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.fc  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        return self.fc(self.enc(self.proj(x)).mean(1)).squeeze(-1)

print("  Training Transformer...")
m = TransformerModel().to(DEVICE)
train_and_predict(m, lr=5e-4)
p = train_and_predict(m, lr=5e-4)
evaluate("Transformer", y_test_actual, p)

# ── 10. YOUR MSTN ─────────────────────────────────────────────
print("\n── YOUR MSTN MODEL ────────────────────────────────────────")

class MSTNFixed(nn.Module):
    """MSTN with Sigmoid output for 0-1 scaled prediction"""
    def __init__(self):
        super().__init__()
        self.mstn = MSTN(input_dim=INP, seq_len=SEQ, dropout=0.3)
        self.mstn.head = nn.Sequential(
            nn.Linear(192, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()   # output 0-1 → unscale to traffic_volume
        )
    def forward(self, x):
        return self.mstn(x)

mstn = MSTNFixed().to(DEVICE)
opt  = torch.optim.AdamW(mstn.parameters(), lr=1e-3, weight_decay=1e-4)
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
crit = nn.HuberLoss()
best = float('inf')

print(f"  Parameters: {sum(p.numel() for p in mstn.parameters()):,}")
print(f"  Training for 25 epochs...")

for ep in range(25):
    mstn.train()
    ep_l = 0
    for xb, yb in tr_dl:
        opt.zero_grad()
        loss = crit(mstn(xb.to(DEVICE)), yb.to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(mstn.parameters(), 1.0)
        opt.step()
        ep_l += loss.item()
    ep_l /= len(tr_dl)
    sch.step()
    if ep_l < best:
        best = ep_l
        torch.save(mstn.state_dict(), 'models/mstn_fixed_best.pth')
    if (ep+1) % 5 == 0:
        print(f"  Epoch {ep+1:2d}/25 | Loss: {ep_l:.6f}")

mstn.load_state_dict(torch.load('models/mstn_fixed_best.pth'))
mstn.eval()
with torch.no_grad():
    p_scaled = mstn(torch.FloatTensor(Xte_s).to(DEVICE)).cpu().numpy()
p_final = unscale_y(p_scaled)
evaluate("MSTN-BiLSTM ★ YOURS", y_test_actual, p_final)

# ── 11. ACTUAL vs PREDICTED TABLE ─────────────────────────────
print("\n── ACTUAL vs PREDICTED (20 Real Examples) ─────────────────")
print(f"\n  {'Index':<8} {'Actual Vol':>10} {'MSTN Pred':>10} "
      f"{'Error':>8} {'Actual':^10} {'Predicted':^10} {'✓/✗'}")
print("  " + "-"*70)

def level_emoji(v):
    if v < 1000: return '🟢 LOW'
    if v < 3000: return '🟡 MED'
    if v < 5000: return '🟠 HIGH'
    return '🔴 SEV'

step = len(y_test_actual) // 20
correct = 0
for i in range(0, len(y_test_actual), step)[:20]:
    actual = y_test_actual[i]
    pred   = p_final[i]
    error  = pred - actual
    al     = level_emoji(actual)
    pl     = level_emoji(pred)
    match  = "✓" if al==pl else "✗"
    if al == pl: correct += 1
    print(f"  {i:<8} {int(actual):>10,} {int(pred):>10,} "
          f"{error:>+8.0f} {al:^12} {pl:^12} {match}")

print(f"\n  Congestion Level Match: {correct}/20 ({correct/20*100:.0f}%)")

# ── 12. FINAL TABLE ───────────────────────────────────────────
print("\n" + "="*80)
print(f"  FINAL RESULTS — 80% Train / 20% Test — REAL traffic_volume")
print("="*80)
print(f"  {'Model':<28} {'MAE':>7} {'RMSE':>8} {'R²%':>8} {'ClassAcc':>10}")
print("-"*80)

cats = {
    'DL Models' : ['LSTM','BiLSTM','GRU','CNN-LSTM','Transformer'],
    'Your Model': ['MSTN-BiLSTM ★ YOURS']
}
for cat, models in cats.items():
    print(f"\n  ── {cat} ──")
    for name in models:
        if name not in results: continue
        r = results[name]
        marker = "  ◄ NOVEL MODEL" if "YOURS" in name else ""
        print(f"  {name:<28} {r['MAE']:>7.1f} {r['RMSE']:>8.1f} "
              f"{max(0,r['R2'])*100:>7.2f}% {r['ClassAcc']:>9.1f}%{marker}")

print("\n" + "="*80)
mstn_r = results['MSTN-BiLSTM ★ YOURS']
print(f"  ★ MSTN R²            : {mstn_r['R2']*100:.2f}%")
print(f"  ★ MSTN MAE           : {mstn_r['MAE']:.1f} vehicles/hour")
print(f"  ★ MSTN Congestion Acc: {mstn_r['ClassAcc']:.1f}%")
print("="*80)

# Save
pd.DataFrame({
    'actual_traffic_volume' : y_test_actual,
    'mstn_predicted_volume' : p_final,
    'error'                 : p_final - y_test_actual,
    'actual_level'          : [level_emoji(v) for v in y_test_actual],
    'predicted_level'       : [level_emoji(v) for v in p_final],
}).to_csv('dl_test_results.csv', index=False)
print(f"\n  Saved → dl_test_results.csv")