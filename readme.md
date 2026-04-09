# 💳 Credit Card Fraud Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **INFO 6105 — Data Science Engineering Methods and Tools**  
> Team 5 · Hiteshi Kawadia | Preksha Praveen · Northeastern University

---

## 📌 Project Overview

This project builds a complete end-to-end machine learning pipeline to detect fraudulent credit card transactions. The dataset contains 284,807 real transactions from European cardholders over two days in September 2013, with only 492 fraud cases — a severe 578:1 class imbalance that makes this a genuinely challenging real-world classification problem.

We train and compare four supervised classifiers, handle class imbalance using SMOTE, and evaluate using metrics appropriate for imbalanced data — F1-Score and ROC-AUC rather than accuracy.

---

## 🎯 Key Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| **KNN** ✅ | **0.9710** | 0.7053 | **0.8171** | 0.8999 |
| Random Forest | 0.3798 | 0.8316 | 0.5215 | **0.9812** |
| Logistic Regression | 0.0520 | **0.8737** | 0.0981 | 0.9628 |
| Decision Tree | 0.0558 | 0.8421 | 0.1047 | 0.9374 |

**Recommended model: KNN** — highest F1-Score (0.8171) and precision (0.9710), with only 3 false positives across 56,746 test transactions.

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
│
├── notebook/
│   └── Team_5_Credit_Card_Fraud_Detection.ipynb   # Main Jupyter notebook
│
├── report/
│   └── Team_5_Final_Report.docx                   # Final written report
│
├── presentation/
│   └── Team_5_Presentation.pdf                    # Slide deck
│
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```

> ⚠️ The dataset (`creditcard.csv`) is not included due to file size (~150MB). Download from Kaggle — link below.

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions:** 284,807 total · 492 fraud (0.172%) · 284,315 legitimate
- **Features:** 31 columns — V1–V28 (PCA-transformed), Time, Amount, Class
- **File size:** ~150 MB
- **Missing values:** None
- **Duplicates:** 1,081 removed during preprocessing

After downloading, place `creditcard.csv` in the root directory before running the notebook.

---

## 🔧 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory.

### 4. Run the notebook
```bash
jupyter notebook notebook/Team_5_Credit_Card_Fraud_Detection.ipynb
```

---

## 🔬 Methodology

### Preprocessing Pipeline
The preprocessing steps follow a strict order to prevent data leakage:

1. **Clean** — Remove 1,081 duplicate rows. Derive `Hour` feature from `Time`, then drop raw `Time`.
2. **Split** — 80/20 stratified train-test split before any scaling or resampling.
3. **Scale** — `StandardScaler` on `Amount` and `Hour` only. `fit_transform` on train, `transform` on test.
4. **SMOTE** — Applied to training set only. Balances 578:1 → 1:1 (226,602 samples each class).

### Models
| Model | Approach | Tuning |
|---|---|---|
| Logistic Regression | Linear baseline | GridSearchCV (C parameter) |
| Decision Tree | Interpretable splits | GridSearchCV (depth, leaf size) |
| Random Forest | Ensemble of 50 trees | Fixed params (computational constraints) |
| KNN | Instance-based, k=5 | Fixed params, trained on pre-SMOTE data |

### Evaluation Metrics
Accuracy was excluded due to severe class imbalance. We used:
- **Precision** — what fraction of fraud alerts are genuine?
- **Recall** — what fraction of actual fraud was caught?
- **F1-Score** — harmonic mean of precision and recall (primary metric)
- **ROC-AUC** — discrimination ability across all thresholds

---

## 📈 Key Findings

- **KNN** achieved the best F1-Score (0.8171) and precision (0.9710) — only 3 false positives in 56,746 test transactions
- **Random Forest** achieved the best ROC-AUC (0.9812) — best threshold-flexible option for high-risk scenarios
- **Logistic Regression** achieved the best recall (0.8737) — catches the most fraud but with many false alarms
- **Decision Tree** root split on `V14` independently confirmed EDA correlation findings and Random Forest feature importance — all three methods identified the same top features

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE oversampling |

---

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
jupyter
```

Install all with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

---

## 👥 Team

| Name | Role |
|---|---|
| Hiteshi Kawadia | Data preprocessing, model building, evaluation |
| Preksha Praveen | EDA, model building, report and presentation |

---

## 📚 References

1. Dal Pozzolo et al. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI*.
2. Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR, 16*, 321–357.
3. Kaggle Dataset: [Credit Card Fraud Detection — MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
4. Scikit-learn documentation: https://scikit-learn.org
5. Imbalanced-learn documentation: https://imbalanced-learn.org

---

## 📄 License

This project is for academic purposes only as part of INFO 6105 at Northeastern University.
