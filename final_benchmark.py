# =============================================================
# FINAL BENCHMARK — ML + DL Models
# Proper 80/20 split with shuffle
# No MSTN for now
# =============================================================
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
from sklearn.model_selection import train_test_split
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
print(f"Rows: {len(df)}")

df['date_time']  = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)
df['hour']       = df['date_time'].dt.hour
df['dow']        = df['date_time'].dt.dayofweek
df['month']      = df['date_time'].dt.month
df['quarter']    = df['date_time'].dt.quarter
df['is_weekend'] = (df['dow'] >= 5).astype(float)
df['is_holiday'] = (df['holiday'] != 'None').astype(float)
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

FEATURES = ['hour','dow','month','quarter','is_weekend','is_holiday',
            'temp_c','rain_1h','snow_1h','clouds_all','weather_enc',
            'rush_hour','night','hour_sin','hour_cos',
            'dow_sin','dow_cos','month_sin','month_cos']
TARGET = 'traffic_volume'

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32)

# ── 2. 80/20 SPLIT WITH SHUFFLE ──────────────────────────────
# Shuffle = True kyunki traffic patterns consistent hain
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train).astype(np.float32)
X_te_s = scaler.transform(X_test).astype(np.float32)

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train traffic_volume: {y_train.min():.0f} → {y_train.max():.0f}")
print(f"Test  traffic_volume: {y_test.min():.0f} → {y_test.max():.0f}")

results = {}

def level(v):
    if v < 1000: return 'LOW'
    if v < 3000: return 'MEDIUM'
    if v < 5000: return 'HIGH'
    return 'SEVERE'

def evaluate(name, actual, predicted, category='ML'):
    actual    = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual-predicted)/(actual+1e-6)))*100
    al   = [level(v) for v in actual]
    pl   = [level(v) for v in predicted]
    cacc = np.mean([a==b for a,b in zip(al,pl)])*100
    results[name] = {
        'MAE':mae,'RMSE':rmse,'R2':r2,
        'MAPE':mape,'ClassAcc':cacc,
        'preds':predicted,'cat':category
    }
    print(f"  ✓ {name:<28} "
          f"MAE={mae:7.1f}  "
          f"RMSE={rmse:7.1f}  "
          f"R²={max(0,r2)*100:.2f}%  "
          f"ClassAcc={cacc:.1f}%")

# ── 3. ML MODELS ─────────────────────────────────────────────
print("\n── ML MODELS ───────────────────────────────────────────────")

m = LinearRegression().fit(X_tr_s, y_train)
evaluate("Linear Regression", y_test, m.predict(X_te_s))

m = Ridge(alpha=1.0).fit(X_tr_s, y_train)
evaluate("Ridge Regression", y_test, m.predict(X_te_s))

m = Lasso(alpha=0.1).fit(X_tr_s, y_train)
evaluate("Lasso Regression", y_test, m.predict(X_te_s))

m = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_train, y_train)
evaluate("Decision Tree", y_test, m.predict(X_test))

m = KNeighborsRegressor(n_neighbors=10, n_jobs=-1).fit(X_tr_s, y_train)
evaluate("KNN (k=10)", y_test, m.predict(X_te_s))

print("  Training Random Forest...")
m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
m.fit(X_train, y_train)
evaluate("Random Forest", y_test, m.predict(X_test))

print("  Training Extra Trees...")
m = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
m.fit(X_train, y_train)
evaluate("Extra Trees", y_test, m.predict(X_test))

m = AdaBoostRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
evaluate("AdaBoost", y_test, m.predict(X_test))

print("  Training Gradient Boosting...")
m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                               random_state=42)
m.fit(X_train, y_train)
evaluate("Gradient Boosting", y_test, m.predict(X_test))

print("  Training XGBoost...")
m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                      max_depth=6, random_state=42, n_jobs=-1)
m.fit(X_train, y_train, verbose=False)
evaluate("XGBoost", y_test, m.predict(X_test))

print("  Training LightGBM...")
m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                       random_state=42, verbose=-1)
m.fit(X_train, y_train)
evaluate("LightGBM", y_test, m.predict(X_test))

# ── 4. DL SETUP ───────────────────────────────────────────────
print("\n── DL MODELS ───────────────────────────────────────────────")

y_min = float(y_train.min())
y_max = float(y_train.max())

def scale_y(arr):   return (arr - y_min) / (y_max - y_min + 1e-8)
def unscale_y(arr): return arr * (y_max - y_min + 1e-8) + y_min

def make_seq(X, y, s):
    Xs, ys = [], []
    for i in range(len(X) - s):
        Xs.append(X[i:i+s])
        ys.append(y[i+s])
    return np.array(Xs, np.float32), np.array(ys, np.float32)

Xtr_seq, ytr_seq = make_seq(X_tr_s, scale_y(y_train), SEQ)
Xte_seq, yte_seq = make_seq(X_te_s, scale_y(y_test),  SEQ)
y_test_dl        = y_test[SEQ:]

print(f"Train sequences: {len(Xtr_seq):,}")
print(f"Test  sequences: {len(Xte_seq):,}")

tr_dl = DataLoader(
    TensorDataset(torch.FloatTensor(Xtr_seq),
                  torch.FloatTensor(ytr_seq)),
    batch_size=64, shuffle=True)

INP = X_tr_s.shape[1]

def train_model(model, epochs=15, lr=1e-3):
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss()
    best_loss = float('inf')
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
            best_loss = ep_l
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)

def get_preds(model):
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(Xte_seq).to(DEVICE)).cpu().numpy()
    return unscale_y(p.flatten())

# LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INP,128,2,batch_first=True,dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(128,64),nn.ReLU(),
                                   nn.Dropout(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x):
        o,_ = self.lstm(x); return self.fc(o[:,-1]).squeeze(-1)

print("  Training LSTM...")
m = LSTMModel().to(DEVICE)
train_model(m, epochs=20)
evaluate("LSTM", y_test_dl, get_preds(m), 'DL')

# BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INP,64,2,batch_first=True,
                            bidirectional=True,dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(128,64),nn.ReLU(),
                                   nn.Dropout(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x):
        o,_ = self.lstm(x); return self.fc(o[:,-1]).squeeze(-1)

print("  Training BiLSTM...")
m = BiLSTMModel().to(DEVICE)
train_model(m, epochs=20)
evaluate("BiLSTM", y_test_dl, get_preds(m), 'DL')

# GRU
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INP,128,2,batch_first=True,dropout=0.2)
        self.fc  = nn.Sequential(nn.Linear(128,64),nn.ReLU(),
                                  nn.Dropout(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, x):
        o,_ = self.gru(x); return self.fc(o[:,-1]).squeeze(-1)

print("  Training GRU...")
m = GRUModel().to(DEVICE)
train_model(m, epochs=20)
evaluate("GRU", y_test_dl, get_preds(m), 'DL')

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
        o,_ = self.lstm(x); return self.fc(o[:,-1]).squeeze(-1)

print("  Training CNN-LSTM...")
m = CNNLSTMModel().to(DEVICE)
train_model(m, epochs=20)
evaluate("CNN-LSTM", y_test_dl, get_preds(m), 'DL')

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
train_model(m, epochs=20, lr=5e-4)
evaluate("Transformer", y_test_dl, get_preds(m), 'DL')

# ── 5. FINAL TABLE ────────────────────────────────────────────
print("\n" + "="*80)
print(f"  FINAL RESULTS — 80/20 Split — Real traffic_volume Test")
print("="*80)
print(f"  {'Model':<28} {'MAE':>7} {'RMSE':>8} {'R²%':>8} {'ClassAcc':>10}")
print("-"*80)

cats = {
    'ML Models': ['Linear Regression','Ridge Regression','Lasso Regression',
                  'Decision Tree','KNN (k=10)','Random Forest','Extra Trees',
                  'AdaBoost','Gradient Boosting','XGBoost','LightGBM'],
    'DL Models': ['LSTM','BiLSTM','GRU','CNN-LSTM','Transformer']
}

for cat, models in cats.items():
    print(f"\n  ── {cat} ──")
    for name in models:
        if name not in results: continue
        r = results[name]
        print(f"  {name:<28} {r['MAE']:>7.1f} {r['RMSE']:>8.1f} "
              f"{max(0,r['R2'])*100:>7.2f}% {r['ClassAcc']:>9.1f}%")

print("\n" + "="*80)
best_r2  = max(results, key=lambda k: results[k]['R2'])
best_mae = min(results, key=lambda k: results[k]['MAE'])
print(f"  🏆 Best R²  : {best_r2} ({results[best_r2]['R2']*100:.2f}%)")
print(f"  🏆 Best MAE : {best_mae} ({results[best_mae]['MAE']:.1f})")
print("="*80)

pd.DataFrame([
    {'Model':k, 'MAE':v['MAE'], 'RMSE':v['RMSE'],
     'R2_percent':max(0,v['R2'])*100, 'ClassAcc':v['ClassAcc']}
    for k,v in results.items()
]).to_csv('benchmark_results.csv', index=False)
print("\n  Saved → benchmark_results.csv")