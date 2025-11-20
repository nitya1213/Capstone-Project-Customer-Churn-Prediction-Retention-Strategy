# 📊 Customer Churn Prediction & Analysis

*A Data-Driven Approach to Understanding User Drop-Offs & Optimizing Retention*

---

## 🚀 Project Overview

This project performs **Customer Churn Analysis and Prediction** using **data preprocessing, EDA, churn-based case studies, and machine learning models**.
The aim is to identify key drivers of churn, predict high-risk customers, and enable strong retention strategies.

🔍 **Key Deliverables**

* 📌 Churn-related case studies & visual insights (EDA)
* ⚙️ Data preprocessing & feature transformation
* 🤖 Model training & evaluation (Logistic Regression, Decision Tree, KNN)
* 🔎 Hyperparameter tuning via GridSearchCV
* 💾 Best model saved (`models/best_model.pkl`)
* 📈 Visuals of churn trends and confusion matrices (`outputs/figures/`)

---

## 📁 Project Structure

```
customer_churn_project
├── data/                      # Input dataset
│   └── customer_churn.csv
├── src/                       # Source code
│   ├── eda_cases.py           # EDA & visualization
│   ├── preprocessing.py       # Data cleaning & transformation
│   ├── tuning.py              # Model hyperparameter optimization
│   ├── modeling.py            # Final training & evaluation
│   ├── train_model.py         # (internal use)
│   └── utils.py               # Helper functions
├── outputs/
│   └── figures/               # EDA & model output visualizations
├── models/                    # Final model & encoders
│   ├── best_model.pkl
│   ├── encoders.pkl
│   ├── encoders_Country.pkl
│   ├── encoders_Membership Status.pkl
│   ├── encoders_Product Purchased.pkl
│   └── encoders_scaler.pkl
├── notebooks/
│   └── 01_EDA_and_CaseStudies.ipynb
├── requirements.txt           # Dependencies
└── run_all.sh                 # Full pipeline automation
```

---

## ⚙️ Setup & Run Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/nitya1213/Capstone-Project-Customer-Churn-Prediction-Retention-Strategy
cd customer_churn_project
```

### 2️⃣ Give execution permission

```bash
chmod +x run_all.sh
```

### 3️⃣ Run the complete pipeline

```bash
./run_all.sh
```

This will:
✔ Create & activate `venv` (if not exists)
✔ Install only missing dependencies
✔ Run EDA → hyperparameter tuning → model training
✔ Save results in `outputs/` & `models/`

---

## 📦 Virtual Environment

The project uses a **virtual environment (`venv/`)** to ensure consistent dependencies.

If you prefer to set up manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔍 Model Performance Summary

| Model               | Accuracy | F1 Score  | Notes                              |
| ------------------- | -------- | --------- | ---------------------------------- |
| Logistic Regression | 0.54     | **0.589** | **Selected (best CV performance)** |
| Decision Tree       | 0.58     | 0.588     | Slightly better accuracy           |
| KNN                 | 0.58     | 0.562     | Moderate                           |

> **Logistic Regression was selected as the final model due to highest cross-validation F1 score, ensuring better generalization despite slight accuracy trade-off.**

Final model is saved at:
📁 `models/best_model.pkl`

---

## 📈 Visual Outputs

Located in: `outputs/figures/`

✔ Churn distribution

✔ Age & country-wise churn

✔ Feedback vs churn

✔ Login frequency & recency trends

✔ Support call influence

✔ Confusion matrices for all models

---

## 🔧 Requirements

* Python **3.8+**
* Recommended: Unix/Linux or WSL
* RAM ≥ 4GB

Install dependencies manually if needed:

```bash
pip install -r requirements.txt
```

---

## 🧪 Reproducibility Checklist

✔ Source code version-controlled
✔ Data preprocessing fully automated
✔ Trained models and encoders exported
✔ Smart dependency handling in `run_all.sh`

---

## 🧠 Possible Future Enhancements

🔹 Add Random Forest or Gradient Boosting models
🔹 Model threshold optimization
🔹 Customer segmentation-based retention strategies
🔹 Flask/FastAPI deployment
🔹 Integration with Tableau / PowerBI dashboards

---

## 👤 Author

**Nitya Rai**
📍 Capstone Project — Customer Churn Prediction & Retention Strategy

🗓 Timeline: August 2025

💡 Focused on data analytics, predictive modeling & business impact

---

## 🏁 Final Notes

To reproduce full results, simply run:

```bash
./run_all.sh
```

All EDA and model outputs will be visible in `outputs/` & `models/`.

---

*Thank you for exploring this project!*
Feel free to fork, raise issues, or suggest enhancements.

---
