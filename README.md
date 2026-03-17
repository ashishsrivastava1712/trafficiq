

# 🚦 TrafficIQ — ML-Powered Urban Congestion Intelligence

TrafficIQ is an **end-to-end Machine Learning system** that predicts urban traffic congestion using historical data, engineered features, and a highly optimized XGBoost model. It delivers **real-time insights and 24-hour forecasts** via a fast Flask API and an interactive dashboard.

---

## 📑 Table of Contents

* [Quick Start](#-quick-start)
* [Project Overview](#-project-overview)
* [System Architecture](#-system-architecture)
* [Data Pipeline & Engineering](#-data-pipeline--engineering)
* [Model Performance](#-model-performance)
* [API Reference](#-api-reference)
* [Frontend Visualization](#-frontend-visualization)
* [Data Flow Example](#-data-flow-example)
* [Tech Stack](#-tech-stack)
* [Future Improvements](#-future-improvements)

---

## 🚀 Quick Start

Run the project locally in two simple steps:

```bash
# Step 1: Generate features and train model
python generate_eda_data.py

# Step 2: Start frontend server
python -m http.server 8000
```

👉 Open in browser:

```
http://localhost:8000/eda.html
```

---

## 📊 Project Overview

TrafficIQ uses the **Metro Interstate Traffic Volume dataset** (48,204 hourly records from 2012–2018) to predict traffic flow.

### 🔑 Key Highlights

* ⚙️ **18 Engineered Features** (temporal + weather + cyclical encoding)
* 📈 **High Accuracy Model** (R² = 0.934)
* ⚡ **Low Latency API** (< 50ms response time)
* 📊 **Interactive Dashboard** (5 visualization types)

---

## 🏗️ System Architecture

```
Raw CSV (48,204 records)
        ↓
[1] EDA Pipeline (eda_analysis.py)
        ↓
[2] Feature Engineering + Model Training (api.py)
        ↓
[3] Flask API Backend
        ↓
[4] Frontend Dashboard (Chart.js)
```

### 🔹 Key Design Principles

* Separation of concerns
* Scalable modular pipeline
* Real-time inference support

---

## 🧬 Data Pipeline & Engineering

### 1️⃣ Data Cleaning

* Converted **Kelvin → Celsius**
* Removed invalid data:

  * 0K temperatures
  * Rain > 500mm
  * Snow > 5mm

📌 Baseline:
**Average traffic = 3,260 vehicles/hour**

---

### 2️⃣ Feature Engineering (18 Features)

#### ⏱ Temporal Features

* `hour`, `day_of_week`, `month`, `quarter`
* Flags:

  * `is_weekend`
  * `rush_hour` (7–9 AM, 4–6 PM)
  * `night` (12–6 AM)

#### 🌦 Weather Encoding

* Categorical mapping:

  * Clear → 2
  * Rain → 4
  * etc.

#### 🔄 Cyclical Encoding (Critical Insight)

```python
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

✅ Captures circular nature of time (e.g., 11 PM ≈ 12 AM)

📌 **Top Feature:** `hour_cos` → confirms rush-hour dominance

---

## 🤖 Model Performance

### Model: **XGBoost Regressor**

**Configuration:**

* n_estimators = 300
* learning_rate = 0.05
* max_depth = 6

---

### 📊 Evaluation Metrics

| Metric   | Score | Interpretation          |
| -------- | ----- | ----------------------- |
| R² Score | 0.934 | Explains 93.4% variance |
| MAE      | 286.8 | Avg error ±287 vehicles |
| RMSE     | 432.5 | Error spread            |

📌 At ~3000 vehicles/hour → **~9.6% error = high precision**

---

## 🔌 API Reference

### Base: `http://localhost:5000`

| Endpoint                  | Method | Description           |
| ------------------------- | ------ | --------------------- |
| `/api/overview`           | GET    | Dashboard KPIs        |
| `/api/predict`            | POST   | Single prediction     |
| `/api/forecast24`         | POST   | 24-hour forecast      |
| `/api/patterns`           | GET    | Traffic trends        |
| `/api/heatmap`            | GET    | Hour vs Day heatmap   |
| `/api/weather`            | GET    | Weather impact        |
| `/api/feature_importance` | GET    | Model explainability  |
| `/api/peak_windows`       | GET    | Peak congestion hours |

---

### 📌 Example Request

```json
POST /api/predict
{
  "hour": 8,
  "dow": 1,
  "month": 6,
  "temp_c": 20,
  "rain": 0,
  "snow": 0,
  "clouds": 40,
  "weather": "Clear"
}
```

### 📌 Example Response

```json
{
  "volume": 5764,
  "level": "SEVERE",
  "risk": 85,
  "color": "#ef4444",
  "factors": ["Morning rush hour — high commuter traffic"]
}
```

---

## 🎨 Frontend Visualization

Built using **Chart.js** with a modern dark UI.

### 📊 Dashboard Features

* KPI Cards (avg, peak traffic)
* 24-hour line charts
* Severity distribution (donut chart)
* 7×24 heatmap grid
* Real-time prediction gauge

---

## 🔄 Data Flow Example

1. User inputs:

   * Monday, 8 AM, Clear, 20°C
2. Frontend → POST `/api/predict`
3. Backend:

   * Feature engineering (18 features)
   * Scaling (StandardScaler)
   * Prediction (XGBoost)
4. Output:

   * Volume: 5764
   * Risk: 85% (SEVERE)
5. UI:

   * Gauge updates (Red zone)

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask
* XGBoost
* Scikit-learn
* Pandas / NumPy

### Frontend

* HTML / CSS / JavaScript
* Chart.js

### ML Pipeline

* Feature Engineering
* Model Training
* Serialization (Pickle)

---

## 🚀 Future Improvements

* 🔮 Real-time traffic API integration (Google Maps / IoT sensors)
* 🧠 Deep Learning models (LSTM for time series)
* ☁️ Cloud deployment (AWS / Kubernetes)
* 📱 Mobile dashboard
* 📊 Advanced explainability (SHAP values)

---

## 👨‍💻 Author

**Anant Shrivastava**
**Ashish Srivastava**
---
