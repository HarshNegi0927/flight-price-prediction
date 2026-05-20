# ✈️ Flight Price Prediction using AWS SageMaker

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![AWS](https://img.shields.io/badge/AWS-SageMaker-orange?style=for-the-badge&logo=amazon-aws)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

This project builds an end-to-end **Flight Price Prediction System** using **AWS SageMaker**. The goal is to predict the price of a flight ticket based on various features such as airline, source, destination, stops, duration, and travel date. The model is trained, evaluated, and **deployed on AWS SageMaker for real-time predictions**.

> 💡 This is a complete MLOps-style project — from raw data to a deployed cloud endpoint.

---

## 🎯 Problem Statement

Flight ticket prices fluctuate frequently based on many factors. Travelers struggle to find the best time and conditions to book at an optimal price. This project leverages machine learning to predict flight prices and help users make smarter booking decisions.

---

## 🗂️ Dataset

| Feature | Description |
|--------|-------------|
| `Airline` | Name of the airline carrier |
| `Source` | City from which the flight departs |
| `Destination` | City of arrival |
| `Total_Stops` | Number of stops between source and destination |
| `Journey_Date` | Date of travel |
| `Dep_Time` | Departure time of the flight |
| `Arrival_Time` | Arrival time at destination |
| `Duration` | Total duration of the flight |
| `Price` | ✅ Target Variable — Ticket price in INR |

---

## 🛠️ Tech Stack

| Category | Tools / Technologies |
|---------|---------------------|
| Language | Python 3.8+ |
| Cloud Platform | AWS SageMaker |
| Storage | Amazon S3 |
| ML Libraries | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | SageMaker Endpoint (Real-Time Inference) |
| Notebook Environment | SageMaker Studio / Jupyter |

---

## 🔄 Project Workflow

```
Raw Data
   │
   ▼
Data Collection & Loading
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Data Preprocessing & Feature Engineering
   │
   ▼
Model Selection & Training (on SageMaker)
   │
   ▼
Model Evaluation
   │
   ▼
Model Deployment (SageMaker Endpoint)
   │
   ▼
Real-Time Price Prediction
```

---

## 📊 Exploratory Data Analysis (EDA)

Key insights discovered during EDA:

- ✈️ **Jet Airways Business** class had the highest average prices
- 🛑 Flights with **more stops** tend to be **cheaper** but take longer
- 🕐 **Evening departures** are generally priced higher
- 📅 Prices spike closer to the **journey date**
- 🏙️ **Delhi → Cochin** route had significant price variation

---

## ⚙️ Feature Engineering

- Extracted **Day, Month** from Journey Date
- Extracted **Hour, Minutes** from Departure and Arrival Time
- Converted **Duration** to total minutes
- Applied **Label Encoding** on categorical variables
- Dropped irrelevant columns (`Route`, `Additional_Info`)

---

## 🤖 ML Model

- **Algorithm Used:** Random Forest Regressor
- **Why Random Forest?** Handles non-linear relationships, robust to outliers, works well with mixed data types

### Model Performance

| Metric | Score |
|--------|-------|
| R² Score | ~0.81 |
| MAE | ~1200 INR |
| RMSE | ~1900 INR |

---

## ☁️ AWS SageMaker Deployment

### Steps followed:

1. **Uploaded dataset** to Amazon S3 bucket
2. **Launched SageMaker Notebook Instance** for development
3. **Trained the model** using SageMaker's built-in training job
4. **Saved the model artifact** back to S3
5. **Deployed the model** as a SageMaker real-time endpoint
6. **Invoked the endpoint** for live predictions using `boto3`

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

payload = json.dumps({
    "Airline": "IndiGo",
    "Source": "Banglore",
    "Destination": "New Delhi",
    "Total_Stops": "non-stop",
    "Duration": "2h 50m",
    "Journey_Day": 24,
    "Journey_Month": 3,
    "Dep_Hour": 6,
    "Dep_Min": 0
})

response = runtime.invoke_endpoint(
    EndpointName='flight-price-predictor',
    ContentType='application/json',
    Body=payload
)

result = json.loads(response['Body'].read().decode())
print(f"Predicted Flight Price: ₹{result['predicted_price']}")
```

---

## 📁 Project Structure

```
flight-price-prediction-sagemaker/
│
├── data/
│   └── flight_price.csv              # Raw dataset
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb  # Preprocessing & feature creation
│   ├── 03_Model_Training.ipynb       # Model building & evaluation
│   └── 04_SageMaker_Deploy.ipynb     # AWS deployment notebook
│
├── src/
│   ├── preprocess.py                 # Data preprocessing scripts
│   ├── train.py                      # SageMaker training script
│   └── predict.py                    # Inference / prediction script
│
├── model/
│   └── model.pkl                     # Saved trained model
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/flight-price-prediction-sagemaker.git
cd flight-price-prediction-sagemaker
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Notebooks in Order

```
01_EDA.ipynb → 02_Feature_Engineering.ipynb → 03_Model_Training.ipynb → 04_SageMaker_Deploy.ipynb
```

### 4. Configure AWS Credentials (for SageMaker deployment)

```bash
aws configure
# Enter your AWS Access Key, Secret Key, Region
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
boto3
sagemaker
joblib
```

---

## 🌟 Key Learnings

- ✅ End-to-end ML pipeline from raw data to cloud deployment
- ✅ Feature engineering on datetime and categorical columns
- ✅ Training and deploying models on **AWS SageMaker**
- ✅ Real-time model inference using **SageMaker endpoints**
- ✅ Working with **Amazon S3** for data and model storage
- ✅ Using **boto3** for AWS service interaction

---

## 🔮 Future Improvements

- [ ] Add a web interface using **Streamlit or Flask**
- [ ] Implement **hyperparameter tuning** using SageMaker HPO
- [ ] Use **XGBoost** for potentially better performance
- [ ] Set up **CI/CD pipeline** for automatic retraining
- [ ] Add **model monitoring** with SageMaker Model Monitor

---

## 👤 Author

**Your Name**
- 📧 Email: your.email@gmail.com
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- 🐙 GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---

⭐ **If you found this project helpful, please give it a star!** ⭐