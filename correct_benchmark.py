import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
import xgboost as xgb
import lightgbm as lgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ    = 24
print(f"Device: {DEVICE}")

# ── 1. LOAD ───────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'Metro_Interstate_Traffic_Volume.csv')
df = pd.read_csv(CSV_PATH)
print(f"Rows loaded: {len(df)}")

# DROP holiday column — 48143 NaN tha usme
df = df.drop(columns=['holiday'])

df['date_time']  = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)
df['hour']       = df['date_time'].dt.hour
df['dow']        = df['date_time'].dt.dayofweek
df['month']      = df['date_time'].dt.month
df['quarter']    = df['date_time'].dt.quarter
df['is_weekend'] = (df['dow'] >= 5).astype(float)
df['temp_c']     = df['temp'] - 273.15
df['rush_hour']  = df['hour'].apply(
    lambda h: 1.0 if h in [7,8,9,16,17,18] else 0.0)
df['night']      = df['hour'].apply(
    lambda h: 1.0 if h in range(0,6) else 0.0)
df['hour_sin']   = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']   = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin']    = np.sin(2 * np.pi * df['dow'] / 7)
df['dow_cos']    = np.cos(2 * np.pi * df['dow'] / 7)
df['month_sin']  = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos']  = np.cos(2 * np.pi * df['month'] / 12)

le = LabelEncoder()
df['weather_enc'] = le.fit_transform(df['weather_main']).astype(float)
df = df.dropna().reset_index(drop=True)
print(f"After cleaning: {len(df)} rows")

FEATURES = ['hour','dow','month','quarter','is_weekend',
            'temp_c','rain_1h','snow_1h','clouds_all','weather_enc',
            'rush_hour','night','hour_sin','hour_cos',
            'dow_sin','dow_cos','month_sin','month_cos']
TARGET = 'traffic_volume'

X_all = df[FEATURES].values.astype(np.float32)
y_all = df[TARGET].values.astype(np.float32)

# ── 2. SPLIT ──────────────────────────────────────────────────
n         = len(X_all)
train_end = int(n * 0.8)

scaler = StandardScaler()
scaler.fit(X_all[:train_end])
X_scaled = scaler.transform(X_all).astype(np.float32)

y_min = float(y_all[:train_end].min())
y_max = float(y_all[:train_end].max())

def scale_y(arr):   return (arr - y_min) / (y_max - y_min + 1e-8)
def unscale_y(arr): return arr * (y_max - y_min + 1e-8) + y_min

X_tr_ml = X_scaled[:train_end]
X_te_ml = X_scaled[train_end:]
y_tr_ml = y_all[:train_end]
y_te_ml = y_all[train_end:]

print(f"\nTotal          : {n:,}")
print(f"Train 80%      : {len(X_tr_ml):,}")
print(f"Test  20%      : {len(X_te_ml):,}")
print(f"Train period   : {df['date_time'].iloc[0]} -> {df['date_time'].iloc[train_end-1]}")
print(f"Test  period   : {df['date_time'].iloc[train_end]} -> {df['date_time'].iloc[-1]}")
print(f"Train vol range: {y_tr_ml.min():.0f} -> {y_tr_ml.max():.0f}")
print(f"Test  vol range: {y_te_ml.min():.0f} -> {y_te_ml.max():.0f}")

results = {}

def level(v):
    if v < 1000: return 'LOW'
    if v < 3000: return 'MEDIUM'
    if v < 5000: return 'HIGH'
    return 'SEVERE'

def evaluate(name, actual, predicted, cat='ML'):
    actual    = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    predicted = np.clip(predicted, 0, 10000)
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
    al   = [level(v) for v in actual]
    pl   = [level(v) for v in predicted]
    cacc = np.mean([a==b for a,b in zip(al,pl)]) * 100
    results[name] = {
        'MAE':mae, 'RMSE':rmse, 'R2':r2,
        'MAPE':mape, 'ClassAcc':cacc,
        'preds':predicted, 'cat':cat
    }
    print(f"  ✓ {name:<28} "
          f"MAE={mae:7.1f}  "
          f"RMSE={rmse:7.1f}  "
          f"R²={max(0,r2)*100:.2f}%  "
          f"ClassAcc={cacc:.1f}%")

# ── 3. ML MODELS ─────────────────────────────────────────────
print("\n── ML MODELS ───────────────────────────────────────────────")

m = LinearRegression().fit(X_tr_ml, y_tr_ml)
evaluate("Linear Regression", y_te_ml, m.predict(X_te_ml))

m = Ridge(alpha=1.0).fit(X_tr_ml, y_tr_ml)
evaluate("Ridge Regression", y_te_ml, m.predict(X_te_ml))

m = Lasso(alpha=0.1).fit(X_tr_ml, y_tr_ml)
evaluate("Lasso Regression", y_te_ml, m.predict(X_te_ml))

m = DecisionTreeRegressor(max_depth=10, random_state=42)
m.fit(X_tr_ml, y_tr_ml)
evaluate("Decision Tree", y_te_ml, m.predict(X_te_ml))

m = KNeighborsRegressor(n_neighbors=10, n_jobs=-1).fit(X_tr_ml, y_tr_ml)
evaluate("KNN (k=10)", y_te_ml, m.predict(X_te_ml))

print("  Training Random Forest...")
m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
m.fit(X_tr_ml, y_tr_ml)
evaluate("Random Forest", y_te_ml, m.predict(X_te_ml))

print("  Training Extra Trees...")
m = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
m.fit(X_tr_ml, y_tr_ml)
evaluate("Extra Trees", y_te_ml, m.predict(X_te_ml))

m = AdaBoostRegressor(n_estimators=100, random_state=42).fit(X_tr_ml, y_tr_ml)
evaluate("AdaBoost", y_te_ml, m.predict(X_te_ml))

print("  Training Gradient Boosting...")
m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                               random_state=42)
m.fit(X_tr_ml, y_tr_ml)
evaluate("Gradient Boosting", y_te_ml, m.predict(X_te_ml))

print("  Training XGBoost...")
m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                      max_depth=6, random_state=42, n_jobs=-1)
m.fit(X_tr_ml, y_tr_ml, verbose=False)
evaluate("XGBoost", y_te_ml, m.predict(X_te_ml))

print("  Training LightGBM...")
m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                       random_state=42, verbose=-1)
m.fit(X_tr_ml, y_tr_ml)
evaluate("LightGBM", y_te_ml, m.predict(X_te_ml))

# ── 4. DL SETUP ───────────────────────────────────────────────
print("\n── DL MODELS ───────────────────────────────────────────────")

def make_seq(X, y, s):
    Xs, ys = [], []
    for i in range(len(X) - s):
        Xs.append(X[i:i+s])
        ys.append(y[i+s])
    return np.array(Xs, np.float32), np.array(ys, np.float32)

y_scaled  = scale_y(y_all)
X_seq_all, y_seq_all = make_seq(X_scaled, y_scaled, SEQ)
y_raw_all = y_all[SEQ:]

n_seq     = len(X_seq_all)
seq_split = int(n_seq * 0.8)

Xtr_seq  = X_seq_all[:seq_split]
Xte_seq  = X_seq_all[seq_split:]
ytr_seq  = y_seq_all[:seq_split]
y_te_raw = y_raw_all[seq_split:]

print(f"Total sequences : {n_seq:,}")
print(f"Train sequences : {len(Xtr_seq):,}")
print(f"Test  sequences : {len(Xte_seq):,}")
print(f"Test  vol range : {y_te_raw.min():.0f} -> {y_te_raw.max():.0f}")

tr_dl = DataLoader(
    TensorDataset(torch.FloatTensor(Xtr_seq),
                  torch.FloatTensor(ytr_seq)),
    batch_size=64, shuffle=True)

INP = X_scaled.shape[1]

def train_model(model, epochs=20, lr=1e-3):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss()
    best_loss  = float('inf')
    best_state = None
    for ep in range(epochs):
        model.train()
        ep_l = 0
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss = crit(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_l += loss.item()
        ep_l /= len(tr_dl)
        sch.step()
        if ep_l < best_loss:
            best_loss  = ep_l
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)

def get_preds(model):
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(Xte_seq).to(DEVICE)).cpu().numpy()
    return np.clip(unscale_y(p.flatten()), 0, 10000)

# LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INP,128,2,batch_first=True,dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(128,64),nn.ReLU(),
                                   nn.Dropout(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training LSTM...")
m = LSTMModel().to(DEVICE)
train_model(m)
evaluate("LSTM", y_te_raw, get_preds(m), 'DL')

# BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INP,64,2,batch_first=True,
                            bidirectional=True,dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(128,64),nn.ReLU(),
                                   nn.Dropout(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training BiLSTM...")
m = BiLSTMModel().to(DEVICE)
train_model(m)
evaluate("BiLSTM", y_te_raw, get_preds(m), 'DL')

# GRU
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INP,128,2,batch_first=True,dropout=0.2)
        self.fc  = nn.Sequential(nn.Linear(128,64),nn.ReLU(),
                                  nn.Dropout(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x):
        o,_ = self.gru(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training GRU...")
m = GRUModel().to(DEVICE)
train_model(m)
evaluate("GRU", y_te_raw, get_preds(m), 'DL')

# CNN-LSTM
class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = nn.Sequential(
            nn.Conv1d(INP,64,3,padding=1),nn.ReLU(),
            nn.Conv1d(64,64,3,padding=1),nn.ReLU())
        self.lstm = nn.LSTM(64,64,2,batch_first=True,dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(64,32),nn.ReLU(),
                                   nn.Linear(32,1),nn.Sigmoid())
    def forward(self, x):
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

print("  Training CNN-LSTM...")
m = CNNLSTMModel().to(DEVICE)
train_model(m)
evaluate("CNN-LSTM", y_te_raw, get_preds(m), 'DL')

# Transformer
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(INP,128)
        enc = nn.TransformerEncoderLayer(128,4,256,dropout=0.1,batch_first=True)
        self.enc = nn.TransformerEncoder(enc,2)
        self.fc  = nn.Sequential(nn.Linear(128,64),nn.ReLU(),
                                  nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x):
        return self.fc(self.enc(self.proj(x)).mean(1)).squeeze(-1)

print("  Training Transformer...")
m = TransformerModel().to(DEVICE)
train_model(m, lr=5e-4)
evaluate("Transformer", y_te_raw, get_preds(m), 'DL')

# ── 5. FINAL TABLE ────────────────────────────────────────────
print("\n" + "="*80)
print(f"  FINAL RESULTS — Real traffic_volume Test")
print(f"  Train: 80% | Test: 20% — Model ne test data kabhi nahi dekha!")
print("="*80)
print(f"  {'Model':<28} {'MAE':>7} {'RMSE':>8} {'R2%':>8} {'ClassAcc':>10}")
print("-"*80)

cats = {
    'ML Models': ['Linear Regression','Ridge Regression','Lasso Regression',
                  'Decision Tree','KNN (k=10)','Random Forest','Extra Trees',
                  'AdaBoost','Gradient Boosting','XGBoost','LightGBM'],
    'DL Models': ['LSTM','BiLSTM','GRU','CNN-LSTM','Transformer']
}

for cat, models in cats.items():
    print(f"\n  -- {cat} --")
    for name in models:
        if name not in results: continue
        r = results[name]
        print(f"  {name:<28} {r['MAE']:>7.1f} {r['RMSE']:>8.1f} "
              f"{max(0,r['R2'])*100:>7.2f}% {r['ClassAcc']:>9.1f}%")

print("\n" + "="*80)
best_r2  = max(results, key=lambda k: results[k]['R2'])
best_mae = min(results, key=lambda k: results[k]['MAE'])
print(f"  Best R2  : {best_r2} ({results[best_r2]['R2']*100:.2f}%)")
print(f"  Best MAE : {best_mae} ({results[best_mae]['MAE']:.1f})")
print("="*80)

pd.DataFrame([
    {'Model':k, 'Category':v['cat'],
     'MAE':round(v['MAE'],1),
     'RMSE':round(v['RMSE'],1),
     'R2_percent':round(max(0,v['R2'])*100,2),
     'ClassAcc':round(v['ClassAcc'],1)}
    for k,v in results.items()
]).to_csv('correct_benchmark_results.csv', index=False)
print("\n  Saved -> correct_benchmark_results.csv")