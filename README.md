# Credit Card Fraud Detection (Machine Learning Project)

An end-to-end Machine Learning educational project and model showcase designed for learning fraud analytics, dealing with extreme class imbalance, evaluating realistic performance metrics, and deploying transaction-level risk inference.

---

## Project Overview & Objectives

In real-world financial systems, fraudulent transactions constitute a tiny fraction of total activity. In this dataset, only **0.172%** (492 out of 284,807 transactions) are fraudulent.

Standard machine learning approaches fail in this scenario due to the **"Accuracy Paradox"** (a dummy model that labels all transactions legitimate attains 99.83% accuracy while catching zero fraud).

This project rebuilds the fraud detection workflow into a structured, step-by-step educational **Jupyter Notebook (`credit_card_fraud_detection.ipynb`)** divided into 10 structured sessions with intuitive explanations, mathematical reasoning, visualizations, and comparative model benchmarking.

---

## Project Structure

```text
fraud_detection_project/
│
├── fraud_detection_guide.ipynb        # Clean interactive educational notebook (10 Sessions)
├── credit_card_fraud_detection.ipynb  # Primary educational notebook (10 Sessions)
├── creditcard.csv                     # European cardholder dataset (284,807 rows)
├── generate_notebook.py               # Script used to assemble & generate the notebook
├── fraud_detection_project.py         # Original Python script (retained for reference)
├── README.md                          # Project documentation and guide
└── .venv/                             # Virtual environment with all dependencies & ipykernel
```

---

## Notebook Curriculum: 10 Educational Sessions

The notebook [credit_card_fraud_detection.ipynb](file:///d:/collage/fraud_detection_project/credit_card_fraud_detection.ipynb) is organized into 10 clear sections:

| Session | Title | Focus & Concepts |
| :---: | :--- | :--- |
| **1** | **Problem Understanding & Domain Context** | What is financial fraud, business cost of False Positives vs. False Negatives, and PCA anonymity. |
| **2** | **Environment Setup & Library Imports** | Setting up Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (`SMOTE`), and Seaborn visual styling. |
| **3** | **Data Ingestion & Sanity Checks** | Inspecting shape, data types, checking for missing values or corrupt records. |
| **4** | **Exploratory Data Analysis (EDA)** | Quantifying the 0.17% imbalance, visualizing with log-scale counts and donut charts, inspecting `Time` & `Amount`, and discovering top predictive PCA features. |
| **5** | **Preprocessing & Feature Scaling** | Why PCA features (V1 to V28) are already standardized, and why `RobustScaler` is used on `Amount` to handle heavy transaction outliers. |
| **6** | **Splitting Strategy & SMOTE** | **Preventing Data Leakage**: Why resampling is restricted strictly to training data, stratified 80/20 train-test split, and how SMOTE synthesizes minority points via nearest neighbors. |
| **7** | **Evaluation Metrics Deep Dive** | Dissecting the Confusion Matrix, Precision, Recall, F1-Score, and why **PR-AUC (Precision-Recall AUC)** is the gold standard over ROC-AUC for imbalanced data. |
| **8** | **Model Showcase & Comparative Benchmark** | Training **Logistic Regression** (linear baseline) vs. **Random Forest** (non-linear bagging ensemble), comparing side-by-side confusion matrices and ROC / PR curves. |
| **9** | **Decision Threshold Tuning & Real-Time Prediction** | Implementing a modular `predict_transaction(sample, model, threshold)` function; adjusting probability thresholds to meet risk tolerance. |
| **10** | **Summary & Takeaways** | Synthesis of findings, engineering trade-offs, and tips for college project viva/presentation. |

---

## Key Models & Results Comparison

| Model | Recall (Fraud Caught) | Precision (True Alarm Rate) | F1-Score | ROC-AUC | PR-AUC | Primary Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (SMOTE)** | ~91% | ~6% | ~0.11 | ~0.97 | ~0.72 | Fast, interpretable, catches almost all fraud but produces more false alarms. |
| **Random Forest (SMOTE)** | ~83% - 85% | ~85% - 88% | ~0.84 | ~0.97 | ~0.86+ | Exceptional balance; significantly cuts down false alarms while catching frauds. |

---

## How to Run the Notebook

### Option A: Inside VS Code / Antigravity IDE (Recommended)
1. Open [credit_card_fraud_detection.ipynb](file:///d:/collage/fraud_detection_project/credit_card_fraud_detection.ipynb).
2. Click **Select Kernel** in the top right corner of the editor.
3. Choose **Python Environments...** -> select **Python 3.11 (`.venv`)** located at `.\.venv\Scripts\python.exe`.
4. Click **Run All** or execute cell by cell to explore each session.

### Option B: Running Jupyter Notebook in Browser
In PowerShell / Command Prompt inside this folder:
```powershell
.\.venv\Scripts\activate
jupyter notebook credit_card_fraud_detection.ipynb
```

---

## Viva / Presentation Discussion Points

If you are presenting this project for college evaluation, be sure to highlight:
1. **Why not just measure Accuracy?**
   - *Answer:* In a dataset with 99.83% non-fraud, a naive model that predicts 0 for every transaction gets 99.83% accuracy without catching any fraud.
2. **What is Data Leakage in Resampling?**
   - *Answer:* If SMOTE or feature scaling is applied to the entire dataset before train-test split, information from the test set leaks into the training distribution, causing artificially inflated performance that fails in production.
3. **Why did we use PR-AUC alongside ROC-AUC?**
   - *Answer:* ROC-AUC evaluates True Positive Rate vs False Positive Rate. When True Negatives are vast (~284,000), False Positive Rate remains tiny even with many false alarms. Precision-Recall curves focus only on minority class positive outcomes.
