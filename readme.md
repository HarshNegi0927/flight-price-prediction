# ✈️ Flight Price Prediction — AWS SageMaker + Streamlit

A machine learning web app that predicts Indian domestic flight prices using XGBoost, trained and tuned on AWS SageMaker with a Streamlit frontend.

🔗 **Live Demo:** https://flight-price-prediction-sd57dhtrdq6bvcvyketqh9.streamlit.app/

---

## 📌 Project Overview

This project builds an end-to-end flight price prediction pipeline:
- Feature engineering on raw flight data
- Model training and hyperparameter tuning on **AWS SageMaker**
- **R² score of 0.79** on test set
- Deployed as an interactive web app on **Streamlit Cloud**

---

## 🏗️ Architecture

```
Raw Data (CSV)
     │
     ▼
Feature Engineering (Scikit-learn Pipeline)
     │
     ▼
AWS S3 (Preprocessed Data Upload)
     │
     ▼
AWS SageMaker — XGBoost Training
     │
     ▼
Bayesian Hyperparameter Tuning (20 jobs)
     │
     ▼
Best Model → xgboost-model.json
     │
     ▼
Streamlit Web App (Local + Cloud)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Cloud Training | AWS SageMaker |
| Cloud Storage | AWS S3 |
| ML Model | XGBoost |
| Feature Engineering | Scikit-learn, Feature-engine |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |
| Language | Python 3.10 |

---

## 🔧 Feature Engineering

The preprocessing pipeline handles:

| Feature | Transformations Applied |
|---|---|
| `airline` | Rare label encoding → One-hot encoding |
| `date_of_journey` | Extract month, week, day of week, day of year → MinMax scale |
| `source` / `destination` | Mean encoding → Power transform + is_north flag |
| `dep_time` / `arrival_time` | Hour/minute extract + part of day + **peak hour flag** |
| `duration` | **RBF similarity (5 percentile anchors)** + duration category + StandardScaler |
| `total_stops` | is_direct_flight flag |
| `additional_info` | Rare label encoding → One-hot + have_info flag |

**Key improvements over baseline:**
- 5 RBF percentile anchors (10/25/50/75/90) instead of 3
- Peak hour flag for departure and arrival times
- Log1p target transform for better model conditioning
- Feature selector threshold lowered to 0.05

---

## ☁️ AWS SageMaker Training

- **Instance:** `ml.m5.xlarge` with Spot Instances
- **Algorithm:** XGBoost 1.2-1 (built-in SageMaker container)
- **Objective:** `reg:squarederror` on log1p(price)
- **Tuning Strategy:** Bayesian Optimization
- **Jobs:** 20 total, 2 parallel
- **Metric:** Minimize `validation:rmse`

**Hyperparameter search space:**

| Parameter | Range |
|---|---|
| `eta` | 0.01 – 0.2 |
| `max_depth` | 4 – 8 |
| `min_child_weight` | 1 – 10 |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |
| `alpha` (L1) | 0 – 2 |
| `lambda` (L2) | 0 – 2 |
| `gamma` | 0 – 5 |

---

## 📊 Model Performance

| Split | R² Score |
|---|---|
| Train | 0.91 |
| Validation | 0.79 |
| **Test** | **0.79** |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/HarshNegi0927/flight-price-prediction
cd flight-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
flight-price-prediction/
│
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── xgboost-model.json      # Trained XGBoost model (JSON format)
│
├── Data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
└── TrainingData_Improved.ipynb   # SageMaker training notebook
```

---

## 📝 Notes

- Model is saved in **JSON format** (version-independent, production safe)
- Target variable `price` was **log1p transformed** during training; predictions are inverse-transformed with `expm1` at inference
- Preprocessor is fit fresh at app startup on `train.csv`
