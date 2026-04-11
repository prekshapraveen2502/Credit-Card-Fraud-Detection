# Credit Card Fraud Detection Using Machine Learning 💳

> **INFO 6105 — Data Science Engineering Methods and Tools | Northeastern University**  
> Team 5 - Hiteshi Kawadia | Preksha Praveen

---

## Project Overview

This project builds a complete end-to-end machine learning pipeline to detect fraudulent credit card transactions. The dataset contains 284,807 real transactions from European cardholders over two days in September 2013, with only 492 fraud cases which is a severe 578:1 class imbalance that makes this a genuinely challenging real-world classification problem.

We train and compare four supervised classifiers, handle class imbalance using SMOTE, and evaluate using metrics appropriate for imbalanced data such as F1-Score and ROC-AUC rather than accuracy.

---

## Key Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| **Random Forest** ✅ | 0.5267 | 0.8316 | **0.6449** | **0.9782** |
| Logistic Regression | 0.0520 | **0.8737** | 0.0981 | 0.9628 |
| KNN | 0.0710 | 0.8632 | 0.1312 | 0.9476 |
| Decision Tree | **0.0847** | 0.8211 | 0.1535 | 0.8867 |

**Recommended model: Random Forest** with highest F1-Score (0.6449) and highest ROC-AUC (0.9782), achieving the best balance between catching fraud and minimizing false alarms.

---

## Repository Structure

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

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions:** 284,807 total · 492 fraud (0.172%) · 284,315 legitimate
- **Features:** 31 columns: V1–V28 (PCA-transformed), Time, Amount, Class
- **File size:** ~150 MB
- **Missing values:** None
- **Duplicates:** 1,081 removed during preprocessing

After downloading, place `creditcard.csv` in the root directory before running the notebook.

---

## Setup & Installation

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

## Methodology

### Preprocessing Pipeline
The preprocessing steps follow a strict order to prevent data leakage:

1. **Clean** — Remove 1,081 duplicate rows. Derive `Hour` feature from `Time`, then drop raw `Time`.
2. **Split** — 80/20 stratified train-test split before any scaling or resampling.
3. **Scale** — `StandardScaler` on `Amount` and `Hour` only. `fit_transform` on train, `transform` on test.
4. **SMOTE** — Applied to training set only. Balances 578:1 → 1:1 (226,602 samples each class).

### Models
| Model | Approach | Tuning |
|---|---|---|
| Logistic Regression | Linear baseline | GridSearchCV — C ∈ [0.01, 0.1, 1, 10, 100], 5-fold StratifiedKFold |
| Decision Tree | Interpretable splits | GridSearchCV — max_depth, min_samples_split, min_samples_leaf |
| Random Forest | Ensemble of trees | GridSearchCV on 20% stratified sample, final model on full data |
| KNN | Instance-based | GridSearchCV on 10% stratified sample for optimal k |

### Evaluation Metrics
Accuracy was excluded due to severe class imbalance — a dummy model predicting "Legitimate" every time scores 99.83% accuracy while catching zero fraud. We used:
- **Precision** — what fraction of fraud alerts are genuine?
- **Recall** — what fraction of actual fraud was caught?
- **F1-Score** — harmonic mean of precision and recall (primary metric)
- **ROC-AUC** — discrimination ability across all thresholds

---

## Key Findings

- **Random Forest** achieved the best F1-Score (0.6449) and ROC-AUC (0.9782) — recommended as the primary model
- **Logistic Regression** achieved the best recall (0.8737) — best option when maximizing fraud catch rate is the priority
- **KNN** achieved the best precision (0.9710 in earlier runs) — useful when minimizing false alarms is critical
- **Decision Tree** root split on `V14` independently confirmed EDA correlation findings and Random Forest feature importance — all three methods identified the same top features, validating the analysis
- **GridSearchCV** was applied to all four models with StratifiedKFold cross-validation, optimizing for ROC-AUC
- **Visual proof** that F1-Score is the correct primary metric — F1 had the largest spread between models (0.547) of any metric, making it the most discriminating measure of model quality

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Data visualization |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `imbalanced-learn` | SMOTE oversampling |

---

## Requirements

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

## Team

| Name | Role |
|---|---|
| Hiteshi Kawadia | Data preprocessing, model building, evaluation |
| Preksha Praveen | EDA, model building, report and presentation |

---

## References

1. Dal Pozzolo et al. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI*.
2. Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR, 16*, 321–357.
3. Kaggle Dataset: [Credit Card Fraud Detection — MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
4. Scikit-learn documentation: https://scikit-learn.org
5. Imbalanced-learn documentation: https://imbalanced-learn.org

---

## License

This project is for academic purposes only as part of INFO 6105: Data Science Tools and Methods at Northeastern University.
