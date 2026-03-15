# =============================================================
# FULL BENCHMARK — ML + DL + YOUR MSTN
# Models: LR, Ridge, SVR, KNN, RF, GBM, XGBoost, LightGBM,
#         MLP, LSTM, BiLSTM, CNN-LSTM, Transformer, MSTN
# =============================================================
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(__file__))
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
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from models.mstn_model import MSTN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ    = 24
print(f"Device: {DEVICE}")
print("="*70)

# ── 1. LOAD & ENGINEER FEATURES ──────────────────────────────
df = pd.read_csv('data/Metro_Interstate_Traffic_Volume.csv')
df['date_time']  = pd.to_datetime(df['date_time'])
df['hour']       = df['date_time'].dt.hour
df['dow']        = df['date_time'].dt.dayofweek
df['month']      = df['date_time'].dt.month
df['year']       = df['date_time'].dt.year
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

results = {}

def evaluate(name, y_true, y_pred, t):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2,
                     'MAPE': mape, 'Time': t}
    print(f"  ✓ {name:<30} MAE={mae:7.1f}  RMSE={rmse:7.1f}  R²={r2:.4f}  MAPE={mape:5.1f}%  [{t:.1f}s]")

print("\n── ML MODELS ──────────────────────────────────────────────")

# Linear Regression
t=time.time(); m=LinearRegression().fit(X_tr_s,y_train)
evaluate("Linear Regression", y_test, m.predict(X_te_s), time.time()-t)

# Ridge
t=time.time(); m=Ridge(alpha=1.0).fit(X_tr_s,y_train)
evaluate("Ridge Regression", y_test, m.predict(X_te_s), time.time()-t)

# Lasso
t=time.time(); m=Lasso(alpha=0.1).fit(X_tr_s,y_train)
evaluate("Lasso Regression", y_test, m.predict(X_te_s), time.time()-t)

# Decision Tree
t=time.time(); m=DecisionTreeRegressor(max_depth=10,random_state=42).fit(X_train,y_train)
evaluate("Decision Tree", y_test, m.predict(X_test), time.time()-t)

# KNN
t=time.time(); m=KNeighborsRegressor(n_neighbors=10,n_jobs=-1).fit(X_tr_s,y_train)
evaluate("KNN (k=10)", y_test, m.predict(X_te_s), time.time()-t)

# SVR
t=time.time(); m=SVR(kernel='rbf',C=100,epsilon=50).fit(X_tr_s[:8000],y_train[:8000])
p_svr = m.predict(X_te_s)
evaluate("SVR (RBF)", y_test, p_svr, time.time()-t)

# Random Forest
print("  Training Random Forest (200 trees)...")
t=time.time(); m=RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1).fit(X_train,y_train)
evaluate("Random Forest", y_test, m.predict(X_test), time.time()-t)
rf_importances = dict(zip(FEATURES, m.feature_importances_))

# Extra Trees
print("  Training Extra Trees...")
t=time.time(); m=ExtraTreesRegressor(n_estimators=200,random_state=42,n_jobs=-1).fit(X_train,y_train)
evaluate("Extra Trees", y_test, m.predict(X_test), time.time()-t)

# AdaBoost
t=time.time(); m=AdaBoostRegressor(n_estimators=100,random_state=42).fit(X_train,y_train)
evaluate("AdaBoost", y_test, m.predict(X_test), time.time()-t)

# Gradient Boosting
print("  Training Gradient Boosting...")
t=time.time(); m=GradientBoostingRegressor(n_estimators=200,learning_rate=0.05,random_state=42).fit(X_train,y_train)
evaluate("Gradient Boosting", y_test, m.predict(X_test), time.time()-t)

# XGBoost
print("  Training XGBoost...")
t=time.time()
m=xgb.XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=6,
                    subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1)
m.fit(X_train,y_train,eval_set=[(X_test,y_test)],verbose=False)
evaluate("XGBoost", y_test, m.predict(X_test), time.time()-t)
xgb_importances = dict(zip(FEATURES, m.feature_importances_))

# LightGBM
print("  Training LightGBM...")
t=time.time()
m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,
                     num_leaves=63,random_state=42,verbose=-1)
m.fit(X_train,y_train)
evaluate("LightGBM", y_test, m.predict(X_test), time.time()-t)

# MLP
print("  Training MLP...")
t=time.time()
m=MLPRegressor(hidden_layer_sizes=(256,128,64),max_iter=200,random_state=42,
               early_stopping=True,learning_rate_init=0.001)
m.fit(X_tr_s,y_train)
evaluate("MLP (256-128-64)", y_test, m.predict(X_te_s), time.time()-t)

# ── Helper: make sequences ────────────────────────────────────
def make_seq(X, y, s):
    Xs, ys = [], []
    for i in range(len(X)-s):
        Xs.append(X[i:i+s]); ys.append(y[i+s])
    return (torch.FloatTensor(np.array(Xs)),
            torch.FloatTensor(np.array(ys)))

Xtr_t, ytr_t = make_seq(X_tr_s, y_train, SEQ)
Xte_t, yte_t = make_seq(X_te_s,  y_test,  SEQ)
tr_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)

y_mean, y_std = float(y_train.mean()), float(y_train.std())

def norm_y(y): return (y - y_mean) / y_std
def denorm_y(y): return y * y_std + y_mean

Xtr_tn, ytr_tn = make_seq(X_tr_s, norm_y(y_train), SEQ)
Xte_tn, yte_tn = make_seq(X_te_s,  norm_y(y_test),  SEQ)
tr_dl_n = DataLoader(TensorDataset(Xtr_tn, ytr_tn), batch_size=64, shuffle=True)

def train_dl_model(model, dl, epochs=15, lr=1e-3, norm=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.HuberLoss()
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            opt.zero_grad()
            crit(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

def eval_dl_model(model, Xte, yte_raw, norm=False):
    model.eval()
    with torch.no_grad():
        p = model(Xte.to(DEVICE)).cpu().numpy()
    if norm: p = denorm_y(p)
    return p, yte_raw

print("\n── DL MODELS ──────────────────────────────────────────────")
INP = X_tr_s.shape[1]

# ── Vanilla LSTM ─────────────────────────────────────────────
class VanillaLSTM(nn.Module):
    def __init__(self, inp, h=128):
        super().__init__()
        self.lstm = nn.LSTM(inp, h, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        o,_ = self.lstm(x); return self.fc(o[:,-1]).squeeze(-1)

t=time.time(); m=VanillaLSTM(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("Vanilla LSTM", yt, p, time.time()-t)

# ── BiLSTM ───────────────────────────────────────────────────
class BiLSTMModel(nn.Module):
    def __init__(self, inp, h=64):
        super().__init__()
        self.lstm = nn.LSTM(inp, h, 2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.fc   = nn.Sequential(nn.Linear(h*2,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        o,_ = self.lstm(x); return self.fc(o[:,-1]).squeeze(-1)

t=time.time(); m=BiLSTMModel(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("BiLSTM", yt, p, time.time()-t)

# ── Stacked LSTM ─────────────────────────────────────────────
class StackedLSTM(nn.Module):
    def __init__(self, inp, h=128):
        super().__init__()
        self.lstm = nn.LSTM(inp, h, 4, batch_first=True, dropout=0.3)
        self.fc   = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1))
    def forward(self, x):
        o,_ = self.lstm(x); return self.fc(o[:,-1]).squeeze(-1)

t=time.time(); m=StackedLSTM(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("Stacked LSTM (4L)", yt, p, time.time()-t)

# ── GRU ──────────────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, inp, h=128):
        super().__init__()
        self.gru = nn.GRU(inp, h, 2, batch_first=True, dropout=0.2)
        self.fc  = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        o,_ = self.gru(x); return self.fc(o[:,-1]).squeeze(-1)

t=time.time(); m=GRUModel(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("GRU", yt, p, time.time()-t)

# ── BiGRU ─────────────────────────────────────────────────────
class BiGRUModel(nn.Module):
    def __init__(self, inp, h=64):
        super().__init__()
        self.gru = nn.GRU(inp, h, 2, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.fc  = nn.Sequential(nn.Linear(h*2,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):
        o,_ = self.gru(x); return self.fc(o[:,-1]).squeeze(-1)

t=time.time(); m=BiGRUModel(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("BiGRU", yt, p, time.time()-t)

# ── CNN-1D ────────────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(inp,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x):
        return self.fc(self.net(x.transpose(1,2)).squeeze(-1)).squeeze(-1)

t=time.time(); m=CNN1D(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("CNN-1D", yt, p, time.time()-t)

# ── CNN-LSTM ──────────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.cnn  = nn.Sequential(
            nn.Conv1d(inp,64,3,padding=1), nn.ReLU(),
            nn.Conv1d(64,64,3,padding=1),  nn.ReLU())
        self.lstm = nn.LSTM(64, 64, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

t=time.time(); m=CNNLSTMModel(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("CNN-LSTM", yt, p, time.time()-t)

# ── CNN-BiLSTM ────────────────────────────────────────────────
class CNNBiLSTM(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.cnn    = nn.Sequential(
            nn.Conv1d(inp,64,3,padding=1), nn.ReLU(),
            nn.Conv1d(64,64,3,padding=1),  nn.ReLU())
        self.bilstm = nn.LSTM(64, 64, 2, batch_first=True,
                               bidirectional=True, dropout=0.2)
        self.fc     = nn.Linear(128, 1)
    def forward(self, x):
        x = self.cnn(x.transpose(1,2)).transpose(1,2)
        o,_ = self.bilstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

t=time.time(); m=CNNBiLSTM(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("CNN-BiLSTM", yt, p, time.time()-t)

# ── Temporal CNN (TCN-style) ──────────────────────────────────
class TCN(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(inp,64,3,padding=2,dilation=2),  nn.ReLU(),
            nn.Conv1d(64,64,3,padding=4,dilation=4),   nn.ReLU(),
            nn.Conv1d(64,64,3,padding=8,dilation=8),   nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(64,1)
    def forward(self, x):
        return self.fc(self.layers(x.transpose(1,2)).squeeze(-1)).squeeze(-1)

t=time.time(); m=TCN(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("TCN (Dilated CNN)", yt, p, time.time()-t)

# ── Transformer ───────────────────────────────────────────────
class TransformerModel(nn.Module):
    def __init__(self, inp, d=128, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(inp, d)
        enc = nn.TransformerEncoderLayer(d, heads, d*2, dropout=0.1, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, layers)
        self.fc   = nn.Linear(d, 1)
    def forward(self, x):
        x = self.proj(x)
        x = self.enc(x)
        return self.fc(x.mean(dim=1)).squeeze(-1)

t=time.time(); m=TransformerModel(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, lr=5e-4, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("Transformer", yt, p, time.time()-t)

# ── Transformer + LSTM hybrid ─────────────────────────────────
class TransLSTM(nn.Module):
    def __init__(self, inp, d=128, heads=4):
        super().__init__()
        self.proj = nn.Linear(inp, d)
        enc = nn.TransformerEncoderLayer(d, heads, d*2, dropout=0.1, batch_first=True)
        self.enc  = nn.TransformerEncoder(enc, 2)
        self.lstm = nn.LSTM(d, 64, 1, batch_first=True)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        x = self.enc(self.proj(x))
        o,_ = self.lstm(x)
        return self.fc(o[:,-1]).squeeze(-1)

t=time.time(); m=TransLSTM(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, lr=5e-4, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("Transformer+LSTM", yt, p, time.time()-t)

# ── Attention-LSTM ────────────────────────────────────────────
class AttnLSTM(nn.Module):
    def __init__(self, inp, h=128):
        super().__init__()
        self.lstm = nn.LSTM(inp, h, 2, batch_first=True, dropout=0.2)
        self.attn = nn.Linear(h, 1)
        self.fc   = nn.Linear(h, 1)
    def forward(self, x):
        o,_ = self.lstm(x)
        w = torch.softmax(self.attn(o), dim=1)
        ctx = (w * o).sum(dim=1)
        return self.fc(ctx).squeeze(-1)

t=time.time(); m=AttnLSTM(INP).to(DEVICE)
train_dl_model(m, tr_dl_n, epochs=15, norm=True)
p,yt = eval_dl_model(m, Xte_tn, denorm_y(yte_tn.numpy()), norm=True)
evaluate("Attention-LSTM", yt, p, time.time()-t)

# ── YOUR MSTN-BiLSTM ──────────────────────────────────────────
print("\n── YOUR NOVEL MODEL ───────────────────────────────────────")
t=time.time()
mstn = MSTN(input_dim=INP, seq_len=SEQ, dropout=0.3).to(DEVICE)
print(f"  MSTN Parameters: {sum(p.numel() for p in mstn.parameters()):,}")
print(f"  Architecture: CNN(k=7,5) + BiLSTM(2L,64) + SGF + SE(r=8) + MHA(4h)")
print(f"  Fusion dim: 192 | O(1) inference via ETA module")

opt  = torch.optim.AdamW(mstn.parameters(), lr=1e-3, weight_decay=1e-4)
sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
crit = nn.HuberLoss()
best_loss = float('inf')

for ep in range(20):
    mstn.train()
    ep_loss = 0
    for xb,yb in tr_dl_n:
        opt.zero_grad()
        loss = crit(mstn(xb.to(DEVICE)), yb.to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(mstn.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    ep_loss /= len(tr_dl_n)
    sch.step()
    if ep_loss < best_loss:
        best_loss = ep_loss
        torch.save(mstn.state_dict(), 'models/mstn_bench_best.pth')
    if (ep+1) % 5 == 0:
        print(f"  Epoch {ep+1:2d}/20 | Loss: {ep_loss:.5f}")

mstn.load_state_dict(torch.load('models/mstn_bench_best.pth'))
mstn.eval()
with torch.no_grad():
    p_n = mstn(Xte_tn.to(DEVICE)).cpu().numpy()
p_final = denorm_y(p_n)
y_final = denorm_y(yte_tn.numpy())
evaluate("MSTN-BiLSTM ★ YOURS", y_final, p_final, time.time()-t)

# ── FINAL COMPARISON TABLE ────────────────────────────────────
print("\n" + "="*85)
print(f"{'COMPLETE MODEL BENCHMARK — TRAFFIC VOLUME PREDICTION':^85}")
print("="*85)
print(f"{'Model':<28} {'MAE':>7} {'RMSE':>8} {'R²':>8} {'Acc%':>7} {'MAPE%':>7} {'Time':>6}")
print("-"*85)

cats = {
    'LINEAR': ['Linear Regression','Ridge Regression','Lasso Regression'],
    'TREE'  : ['Decision Tree','Random Forest','Extra Trees','AdaBoost','Gradient Boosting'],
    'BOOST' : ['XGBoost','LightGBM'],
    'NEURAL': ['MLP (256-128-64)'],
    'RNN'   : ['Vanilla LSTM','BiLSTM','Stacked LSTM (4L)','GRU','BiGRU'],
    'CNN'   : ['CNN-1D','CNN-LSTM','CNN-BiLSTM','TCN (Dilated CNN)'],
    'HYBRID': ['Transformer','Transformer+LSTM','Attention-LSTM'],
    'YOURS' : ['MSTN-BiLSTM ★ YOURS']
}

for cat, models in cats.items():
    print(f"\n  ── {cat} ──")
    for name in models:
        if name not in results: continue
        r = results[name]
        marker = " ◄ YOUR NOVEL MODEL" if "YOURS" in name else ""
        print(f"  {name:<26} {r['MAE']:>7.1f} {r['RMSE']:>8.1f} {r['R2']:>8.4f} "
              f"{max(0,r['R2'])*100:>6.2f}% {r['MAPE']:>6.1f}% {r['Time']:>5.1f}s{marker}")

print("\n" + "="*85)

# Best model per metric
best_r2  = max(results, key=lambda k: results[k]['R2'])
best_mae = min(results, key=lambda k: results[k]['MAE'])
print(f"  🏆 Best R²  : {best_r2}  ({results[best_r2]['R2']*100:.2f}%)")
print(f"  🏆 Best MAE : {best_mae}  ({results[best_mae]['MAE']:.1f})")
print(f"  ★  MSTN R²  : {results['MSTN-BiLSTM ★ YOURS']['R2']*100:.2f}%")
print(f"  ★  MSTN MAE : {results['MSTN-BiLSTM ★ YOURS']['MAE']:.1f}")
print("="*85)

# Save to CSV
rows = [{'Model': k, 'Category': next((c for c,ms in cats.items() if k in ms), 'OTHER'), **v}
        for k,v in results.items()]
pd.DataFrame(rows).round(4).to_csv('benchmark_results.csv', index=False)
print("\n  Results saved to benchmark_results.csv")