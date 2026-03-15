# =============================================================
# PROPER REAL TEST
# Chronological split — model future data pe test hoga
# Actual traffic_volume vs Predicted traffic_volume
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
import lightgbm as lgb
from models.mstn_model import MSTN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ    = 24
print(f"Device: {DEVICE}")

# ── 1. LOAD DATA ──────────────────────────────────────────────
print("\n[1] Loading real dataset...")
df = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')
df['date_time']  = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)  # chronological!

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
TARGET = 'traffic_volume'

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)
dates = df['date_time']

# ── 2. CHRONOLOGICAL SPLIT ────────────────────────────────────
# Train: First 80% (older data)
# Test:  Last 20% (newer data — model ne kabhi nahi dekha!)
n           = len(X)
train_end   = int(n * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_test,  y_test  = X[train_end:], y[train_end:]
dates_test       = dates.iloc[train_end:].reset_index(drop=True)

print(f"\n  Total records : {n:,}")
print(f"  Train period  : {dates.iloc[0]} → {dates.iloc[train_end-1]}")
print(f"  Test period   : {dates.iloc[train_end]} → {dates.iloc[-1]}")
print(f"  Train samples : {len(X_train):,}")
print(f"  Test samples  : {len(X_test):,}")
print(f"\n  ⚠️  Model sirf train period dekha hai!")
print(f"  ✅  Test period = FUTURE data = real world test!")

# ── 3. SCALE ─────────────────────────────────────────────────
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train).astype(np.float32)
X_te_s = scaler.transform(X_test).astype(np.float32)

results = {}

def evaluate(name, actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100

    def level(v):
        if v < 1000:  return 'LOW'
        if v < 3000:  return 'MEDIUM'
        if v < 5000:  return 'HIGH'
        return 'SEVERE'

    al   = [level(v) for v in actual]
    pl   = [level(v) for v in predicted]
    cacc = np.mean([a==b for a,b in zip(al,pl)]) * 100

    results[name] = {
        'MAE': mae, 'RMSE': rmse,
        'R2': r2, 'MAPE': mape,
        'ClassAcc': cacc,
        'predictions': predicted
    }
    marker = " ◄ YOUR MODEL" if "MSTN" in name else ""
    print(f"  ✓ {name:<28} "
          f"MAE={mae:7.1f}  "
          f"RMSE={rmse:7.1f}  "
          f"R²={r2:.4f}  "
          f"ClassAcc={cacc:.1f}%"
          f"{marker}")

# ── 4. ML MODELS ─────────────────────────────────────────────
print("\n[2] Training & Testing ML Models on REAL data...")

# Linear Regression
m = LinearRegression().fit(X_tr_s, y_train)
evaluate("Linear Regression", y_test, m.predict(X_te_s))

# Ridge
m = Ridge(alpha=1.0).fit(X_tr_s, y_train)
evaluate("Ridge Regression", y_test, m.predict(X_te_s))

# Random Forest
print("  Training Random Forest...")
m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
m.fit(X_train, y_train)
evaluate("Random Forest", y_test, m.predict(X_test))

# XGBoost
print("  Training XGBoost...")
m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                      max_depth=6, random_state=42, n_jobs=-1)
m.fit(X_train, y_train, verbose=False)
evaluate("XGBoost", y_test, m.predict(X_test))

# LightGBM
print("  Training LightGBM...")
m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                       random_state=42, verbose=-1)
m.fit(X_train, y_train)
evaluate("LightGBM", y_test, m.predict(X_test))

# ── 5. DL MODELS ─────────────────────────────────────────────
print("\n[3] Training & Testing DL Models on REAL data...")

y_mean = float(y_train.mean())
y_std  = float(y_train.std())

def make_seq(X, y, s):
    Xs, ys = [], []
    for i in range(len(X)-s):
        Xs.append(X[i:i+s])
        ys.append(y[i+s])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# Normalized sequences
Xtr_n, ytr_n = make_seq(X_tr_s, (y_train-y_mean)/y_std, SEQ)
Xte_n, yte_n = make_seq(X_te_s, (y_test -y_mean)/y_std, SEQ)
dates_seq     = dates_test.iloc[SEQ:].reset_index(drop=True)
y_test_seq    = y_test[SEQ:]  # actual traffic_volume for seq test

tr_dl = DataLoader(
    TensorDataset(torch.FloatTensor(Xtr_n),
                  torch.FloatTensor(ytr_n)),
    batch_size=64, shuffle=True)

def train_model(model, epochs=15, lr=1e-3):
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

def predict_dl(model, Xte):
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(Xte).to(DEVICE)).cpu().numpy()
    return p * y_std + y_mean  # denormalize → actual traffic_volume scale

INP = X_tr_s.shape[1]

# LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INP, 128, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training LSTM...")
m = LSTMModel().to(DEVICE)
train_model(m)
evaluate("LSTM", y_test_seq, predict_dl(m, Xte_n))

# BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INP, 64, 2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training BiLSTM...")
m = BiLSTMModel().to(DEVICE)
train_model(m)
evaluate("BiLSTM", y_test_seq, predict_dl(m, Xte_n))

# GRU
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INP, 128, 2, batch_first=True, dropout=0.2)
        self.fc  = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        o,_ = self.gru(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training GRU...")
m = GRUModel().to(DEVICE)
train_model(m)
evaluate("GRU", y_test_seq, predict_dl(m, Xte_n))

# CNN-LSTM
class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = nn.Sequential(
            nn.Conv1d(INP,64,3,padding=1), nn.ReLU(),
            nn.Conv1d(64,64,3,padding=1),  nn.ReLU())
        self.lstm = nn.LSTM(64, 64, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training CNN-LSTM...")
m = CNNLSTMModel().to(DEVICE)
train_model(m)
evaluate("CNN-LSTM", y_test_seq, predict_dl(m, Xte_n))

# Transformer
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(INP, 128)
        enc = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, 2)
        self.fc   = nn.Linear(128, 1)
    def forward(self, x):
        return self.fc(self.enc(self.proj(x)).mean(1)).squeeze(-1)

print("  Training Transformer...")
m = TransformerModel().to(DEVICE)
train_model(m, lr=5e-4)
evaluate("Transformer", y_test_seq, predict_dl(m, Xte_n))

# ── 6. YOUR MSTN ──────────────────────────────────────────────
print("\n[4] Training YOUR MSTN on REAL data...")
mstn = MSTN(input_dim=INP, seq_len=SEQ, dropout=0.3).to(DEVICE)
opt  = torch.optim.AdamW(mstn.parameters(), lr=1e-3, weight_decay=1e-4)
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
crit = nn.HuberLoss()
best = float('inf')

for ep in range(20):
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
        torch.save(mstn.state_dict(), 'models/mstn_proper_best.pth')
    if (ep+1) % 5 == 0:
        print(f"  Epoch {ep+1:2d}/20 | Loss: {ep_l:.5f}")

mstn.load_state_dict(torch.load('models/mstn_proper_best.pth'))
evaluate("MSTN-BiLSTM ★ YOURS", y_test_seq, predict_dl(mstn, Xte_n))

# ── 7. ACTUAL vs PREDICTED TABLE ──────────────────────────────
print("\n[5] Actual vs Predicted — 20 Real Examples:")
print(f"\n  {'Date & Time':<20} {'ACTUAL':>8} {'MSTN Pred':>10} "
      f"{'Error':>8} {'Actual Level':<12} {'Pred Level':<12} {'Match'}")
print("  " + "-"*85)

mstn_preds = results['MSTN-BiLSTM ★ YOURS']['predictions']

def level(v):
    if v < 1000:  return '🟢 LOW'
    if v < 3000:  return '🟡 MEDIUM'
    if v < 5000:  return '🟠 HIGH'
    return '🔴 SEVERE'

# Pick 20 evenly spaced samples
step = len(y_test_seq) // 20
idxs = list(range(0, len(y_test_seq), step))[:20]

correct = 0
for i in idxs:
    actual = y_test_seq[i]
    pred   = mstn_preds[i]
    error  = pred - actual
    al     = level(actual)
    pl     = level(pred)
    match  = "✓" if al==pl else "✗"
    if al == pl: correct += 1
    dt = str(dates_seq.iloc[i])[:16]
    print(f"  {dt:<20} {int(actual):>8,} {int(pred):>10,} "
          f"{error:>+8.0f} {al:<14} {pl:<14} {match}")

print(f"\n  Matched: {correct}/20 ({correct/20*100:.0f}%)")

# ── 8. FINAL RESULTS TABLE ────────────────────────────────────
print("\n" + "="*85)
print(f"  {'FINAL COMPARISON — ALL MODELS — REAL traffic_volume TEST':^83}")
print(f"  {'Train: Past data | Test: FUTURE data model never saw':^83}")
print("="*85)
print(f"  {'Model':<28} {'MAE':>7} {'RMSE':>8} {'R²':>8} "
      f"{'R²%':>7} {'ClassAcc':>10}")
print("-"*85)

categories = {
    'ML Models' : ['Linear Regression','Ridge Regression',
                   'Random Forest','XGBoost','LightGBM'],
    'DL Models' : ['LSTM','BiLSTM','GRU','CNN-LSTM','Transformer'],
    'Your Model': ['MSTN-BiLSTM ★ YOURS']
}

for cat, models in categories.items():
    print(f"\n  ── {cat} ──")
    for name in models:
        if name not in results: continue
        r = results[name]
        marker = "  ◄ YOUR NOVEL MODEL" if "YOURS" in name else ""
        print(f"  {name:<28} {r['MAE']:>7.1f} {r['RMSE']:>8.1f} "
              f"{r['R2']:>8.4f} {max(0,r['R2'])*100:>6.2f}% "
              f"{r['ClassAcc']:>9.1f}%{marker}")

print("\n" + "="*85)
best_r2  = max(results, key=lambda k: results[k]['R2'])
best_mae = min(results, key=lambda k: results[k]['MAE'])
mstn_r   = results['MSTN-BiLSTM ★ YOURS']
print(f"  🏆 Best R²   : {best_r2} ({results[best_r2]['R2']*100:.2f}%)")
print(f"  🏆 Best MAE  : {best_mae} ({results[best_mae]['MAE']:.1f})")
print(f"  ★  MSTN R²   : {mstn_r['R2']*100:.2f}%")
print(f"  ★  MSTN MAE  : {mstn_r['MAE']:.1f} vehicles/hour")
print(f"  ★  MSTN Class: {mstn_r['ClassAcc']:.1f}% congestion accuracy")
print("="*85)

# Save CSV
import pandas as pd
pd.DataFrame({
    'datetime'      : dates_seq.iloc[SEQ:].values,
    'actual_volume' : y_test_seq,
    'mstn_predicted': mstn_preds,
    'error'         : mstn_preds - y_test_seq
}).to_csv('proper_test_results.csv', index=False)
print(f"\n  Results saved → proper_test_results.csv")