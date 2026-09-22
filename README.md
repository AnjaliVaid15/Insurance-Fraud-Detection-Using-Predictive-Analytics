# Insurance Fraud Detection Using Predictive Analytics

A machine learning project that uses insurance claim data to predict fraud risk, prioritize claims for investigation, and provide interpretable risk signals for insurance investigators.

## Project Overview

Insurance fraud creates significant financial and operational challenges for insurers. Manually reviewing every claim is resource-intensive, while relying only on simple rules can miss complex patterns.

This project develops a **predictive analytics pipeline for insurance fraud detection** using Python and machine learning.

The system analyzes claim, policy, incident, customer, and loss-related information to estimate the probability that an insurance claim may be fraudulent.

The objective is not to automatically reject claims. Instead, the model produces a **fraud probability and risk category** that can be used to prioritize claims for further human investigation.

---

## Business Problem

An insurance company receives a large volume of claims, but only a subset may be fraudulent. Investigating every claim equally can increase investigation costs and delay legitimate claims.

The business problem is therefore:

> **How can an insurance company use historical claim data and predictive analytics to identify claims with a higher probability of fraud and prioritize them for investigation?**

### Business Objectives

* Identify patterns associated with fraudulent insurance claims.
* Develop a machine learning classification model.
* Handle class imbalance appropriately.
* Generate fraud-risk probabilities.
* Categorize claims into Low, Medium, and High Risk.
* Explain the factors contributing to model predictions.
* Support investigators in prioritizing suspicious claims.
* Demonstrate how predictive analytics can be integrated into an insurance fraud workflow.

---

## Dataset

The project uses an insurance claims dataset containing **1,000 claim records and 39 variables**.

The target variable is:

`fraud_reported`

where:

* `Y` = Fraud reported
* `N` = Fraud not reported

### Target Distribution

| Class     |    Claims | Percentage |
| --------- | --------: | ---------: |
| Non-Fraud |       753 |      75.3% |
| Fraud     |       247 |      24.7% |
| **Total** | **1,000** |   **100%** |

Because fraudulent claims represent a minority class, the project treats this as an **imbalanced classification problem**.

### Major Data Categories

The dataset contains information related to:

* Policy characteristics
* Customer demographics
* Policy premium
* Policy tenure
* Incident type
* Incident severity
* Incident timing
* Number of vehicles involved
* Bodily injuries
* Witnesses
* Police report availability
* Claim amount
* Injury claim
* Property claim
* Vehicle claim
* Vehicle information
* Customer occupation and hobbies

---

## Analytical Approach

The project follows an end-to-end predictive analytics workflow:

```text
Business Problem
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Train/Test Split
       ↓
Preprocessing
       ↓
Class Imbalance Handling
       ↓
Model Development
       ↓
Model Evaluation
       ↓
Hyperparameter Tuning
       ↓
Model Explainability
       ↓
Fraud Risk Scoring
       ↓
Business Interpretation
```

---

# Project Structure

```text
Insurance-Fraud-Prediction/
│
├── data/
│   ├── raw/
│   │   └── insurance_claims.csv
│   │
│   └── processed/
│       ├── train_engineered.csv
│       ├── test_engineered.csv
│       ├── train_engineered_demo.csv
│       └── test_engineered_demo.csv
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Preprocessing.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Development.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   └── 07_Explainability.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   ├── fraud_model.pkl
│   ├── preprocessor.pkl
│   ├── feature_names.json
│   ├── high_value_threshold.json
│   ├── baseline_model_comparison.csv
│   ├── baseline_vs_tuned.csv
│   └── test_set_risk_scores.csv
│
├── visualizations/
│   ├── 01_target_distribution.png
│   ├── 02_correlation_matrix.png
│   ├── 03_claim_amount_boxplot.png
│   ├── 04_fraud_rate_by_severity.png
│   ├── 05_tenure_distribution.png
│   ├── 06_fraud_rate_by_hobby.png
│   ├── 07_fraud_rate_incidenttype_vehicles.png
│   ├── 08_confusion_matrix.png
│   ├── 09_roc_pr_curves.png
│   ├── 10_feature_importance.png
│   └── 11_shap_summary.png
│
└── requirements.txt
```

---

# Data Preprocessing

Several preprocessing steps were implemented to create a leakage-aware modeling pipeline.

### Missing Values

The dataset contains missing values represented both as:

* `NaN`
* `"?"`

These were standardized before modeling.

Numerical missing values were handled using **median imputation**, while categorical missing values were handled using **most-frequent imputation**.

### Duplicate Records

Exact duplicate claim records were removed.

### Identifier Removal

Variables such as:

* `policy_number`
* `insured_zip`
* `incident_location`

were excluded from modeling because they are identifier/high-cardinality variables rather than generalizable fraud predictors.

Date variables were transformed into meaningful features before being removed from the final modeling matrix.

### Encoding

Categorical variables were transformed using **One-Hot Encoding**.

### Scaling

Numerical variables were standardized using `StandardScaler`.

The preprocessing pipeline was fitted only on the training data and subsequently applied to the test data to reduce the risk of data leakage.

---

# Feature Engineering

Several business-oriented features were created to capture potential fraud signals available at claim-intake time.

### Policy Tenure

```text
policy_tenure_days =
incident_date - policy_bind_date
```

This measures how long the policy had been active before the incident.

### Claim-to-Premium Ratio

```text
claim_to_premium_ratio =
total_claim_amount / policy_annual_premium
```

This measures the size of the claimed loss relative to the annual premium.

### Claim Component Shares

Additional features were created for:

```text
injury_claim_share
property_claim_share
vehicle_claim_share
```

These represent the contribution of each claim component to the total claim amount.

### Tenure Group

Customers were grouped into:

* New (<1 year)
* Established (1–5 years)
* Loyal (5+ years)

### Low Witness Flag

```text
low_witness_flag = 1
```

when no witnesses were reported.

### No Police Report Flag

```text
no_report_flag = 1
```

when a police report was not available.

### High-Value Claim Flag

Claims above the **75th percentile of the training-set claim amount distribution** were flagged as high-value claims.

The threshold was calculated using the training data and reused on the test set to avoid leakage.

### Severity Mismatch

A `severity_vehicle_mismatch` feature was created when:

* Incident severity was classified as trivial/minor
* More than one vehicle/party was involved
* Bodily injuries were reported

This captures a potential inconsistency in the reported incident characteristics.

---

# Handling Class Imbalance

Fraudulent claims represent approximately **24.7%** of the dataset.

Therefore, accuracy alone is not sufficient for evaluating the model.

The project uses **SMOTE on the training data** to address class imbalance.

Importantly, resampling is performed after the train/test split so that synthetic observations do not influence the held-out test set.

The project places particular emphasis on:

* Recall
* Precision
* F1-score
* ROC-AUC
* PR-AUC

with **PR-AUC and fraud recall** receiving particular attention because fraud is the minority class.

---

# Machine Learning Models

The following classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

The models were evaluated on the held-out test set.

---

# Model Performance

The baseline results obtained from the project are:

| Model               | Accuracy | Precision | Recall |    F1 |   ROC-AUC |    PR-AUC |
| ------------------- | -------: | --------: | -----: | ----: | --------: | --------: |
| Gradient Boosting   |    82.5% |     64.0% |  65.3% | 64.6% |     0.841 | **0.598** |
| XGBoost             |    80.5% |     61.4% |  55.1% | 58.1% |     0.829 |     0.586 |
| Random Forest       |    81.5% |     65.8% |  51.0% | 57.5% | **0.848** |     0.563 |
| Decision Tree       |    81.0% |     60.4% |  65.3% | 62.7% |     0.752 |     0.540 |
| Logistic Regression |    81.0% |     60.4% |  65.3% | 62.7% |     0.808 |     0.519 |

The baseline Gradient Boosting model produced a test-set **PR-AUC of approximately 0.598** and **fraud recall of approximately 65.3%**.

For fraud detection, PR-AUC provides useful information about performance on the positive/minority class, while recall indicates how many of the fraudulent claims in the test set were identified.

---

# Hyperparameter Tuning

The Gradient Boosting model was selected for further tuning based on the model-selection logic implemented in the project.

`RandomizedSearchCV` with **5-fold Stratified Cross-Validation** was used.

The hyperparameter search optimized for:

```text
average_precision
```

which corresponds to PR-AUC / Average Precision.

### Baseline vs Tuned Model

| Model                     | Accuracy | Precision | Recall |    F1 | ROC-AUC | PR-AUC |
| ------------------------- | -------: | --------: | -----: | ----: | ------: | -----: |
| Gradient Boosting         |    82.5% |     64.0% |  65.3% | 64.6% |   0.841 |  0.598 |
| Gradient Boosting – Tuned |    80.0% |     60.0% |  55.1% | 57.4% |   0.822 |  0.570 |

The tuned model did not improve the business-relevant test-set metrics sufficiently.

Therefore, the project retains the **baseline Gradient Boosting model** as the final scoring model rather than tuning the model simply for the sake of optimization.

This provides an important practical modeling lesson:

> Hyperparameter tuning does not necessarily improve real-world model performance, and model selection should be based on business-relevant evaluation metrics rather than optimization alone.

---

# Fraud Risk Scoring

The final model generates a probability between 0 and 1 representing the predicted likelihood of fraud.

Claims are then categorized into three risk tiers:

| Fraud Probability | Risk Category |
| ----------------: | ------------- |
|            < 0.30 | Low Risk      |
|       0.30 – 0.59 | Medium Risk   |
|            ≥ 0.60 | High Risk     |

Example output:

```text
Claim ID    Fraud Probability    Risk Category
------------------------------------------------
120         93.50%               High Risk
119         92.50%               High Risk
124         91.61%               High Risk
121         90.60%               High Risk
194         89.97%               High Risk
```

These risk categories are intended for **investigation prioritization**, not automatic claim rejection.

---

# Model Explainability

Fraud detection requires interpretability because investigators and business stakeholders need to understand why a claim has been flagged.

The project includes:

* Feature importance
* SHAP analysis
* Confusion matrix
* ROC/PR curves
* Fraud-risk scoring

Visualizations are available in the `visualizations/` directory.

The explainability workflow is designed to identify which claim and policy characteristics contribute to model predictions.

Importantly, model association does not imply that a particular feature causes fraud.

---

# Business Application

A practical implementation could follow this workflow:

```text
New Insurance Claim
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Fraud Prediction Model
        ↓
Fraud Probability
        ↓
Risk Categorization
        ↓
Investigation Prioritization
        ↓
Human Investigator Review
        ↓
Final Claim Decision
```

For example:

### Low Risk

The claim can proceed through the normal claims workflow, subject to standard controls.

### Medium Risk

The claim can receive additional review or verification depending on investigation capacity.

### High Risk

The claim can be prioritized for detailed investigation.

The model should **not independently reject or deny claims**.

---

# Model Governance and Ethics

Insurance fraud prediction has significant implications for customers and insurers.

The project therefore follows a **human-in-the-loop approach**.

Key considerations include:

* Data privacy
* Explainability
* False positives
* False negatives
* Bias and fairness
* Model drift
* Data drift
* Auditability
* Appropriate use of customer information
* Human investigation before adverse decisions

A high predicted fraud probability should be treated as a **risk signal**, not proof of fraud.

---

# Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn

### Explainable AI

* SHAP

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook

---

# How to Run the Project

## 1. Clone the repository

```bash
git clone https://github.com/your-username/Insurance-Fraud-Prediction.git
cd Insurance-Fraud-Prediction
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Then execute the notebooks in sequence:

```text
01_Data_Understanding.ipynb
02_EDA.ipynb
03_Preprocessing.ipynb
04_Feature_Engineering.ipynb
05_Model_Development.ipynb
06_Model_Evaluation.ipynb
07_Explainability.ipynb
```

---

# Scoring New Claims

The project also includes a prediction script that can score new insurance claims.

Example:

```bash
python src/predict.py --input path/to/new_claims.csv --output path/to/scored_claims.csv
```

The script produces:

```text
claim_id
fraud_probability
risk_category
```

Example:

```text
claim_id | fraud_probability | risk_category
---------------------------------------------
C001     | 0.8245            | High Risk
C002     | 0.4162            | Medium Risk
C003     | 0.0817            | Low Risk
```

---

# Key Project Insights

The project demonstrates several important analytics concepts:

1. **Fraud detection is an imbalanced classification problem.**

2. **Accuracy alone is not sufficient** for evaluating fraud detection models.

3. **PR-AUC and recall** provide useful information when the fraudulent class is relatively uncommon.

4. **Feature engineering can translate domain knowledge into machine-learning variables.**

5. **Data leakage must be controlled carefully**, particularly when creating thresholds and preprocessing features.

6. **Hyperparameter tuning does not guarantee better test-set performance.**

7. **Risk scoring is more operationally useful than a simple binary prediction** because investigators can prioritize claims.

8. **Explainability is important** when machine learning is used in a high-impact insurance workflow.

9. **Human-in-the-loop decision-making** is essential; a predictive model should support investigators rather than independently determine whether a claim is fraudulent.

---

# Limitations

This project has several limitations:

* The dataset contains only 1,000 observations.
* The dataset may not represent the complexity of a production insurance claims environment.
* Fraud patterns can vary across insurance products, regions, customer populations, and time periods.
* Model performance may change when applied to real-world data.
* Historical labels may contain investigation or reporting biases.
* The risk thresholds used in this project are analytical thresholds and would require business validation before production deployment.
* Additional temporal and behavioral claim-history data could improve the modeling framework.
* Production deployment would require extensive model validation, monitoring, governance, and fairness testing.

---

# Future Scope

Potential extensions include:

* Cost-sensitive learning
* Probability calibration
* Threshold optimization based on investigator capacity
* Advanced ensemble models
* Temporal fraud-pattern analysis
* Network analysis for linked claims
* Graph-based fraud detection
* Anomaly detection
* Real-time claim scoring
* Model monitoring
* Data drift monitoring
* SHAP-based investigator dashboards
* Power BI integration
* REST API deployment
* Streamlit-based fraud investigation interface
* Human investigator feedback loops
* Automated model retraining pipelines

---

# Project Outcome

This project demonstrates an end-to-end **predictive analytics solution for insurance fraud detection**, combining:

**Business Problem → Data Analysis → Feature Engineering → Machine Learning → Model Evaluation → Explainable AI → Risk Scoring → Business Decision Support**

The project is designed to demonstrate practical capabilities in **Python, machine learning, predictive analytics, fraud analytics, feature engineering, model evaluation, and business interpretation**.


---

## Disclaimer

This project is developed for educational and portfolio purposes.

The model's predictions represent statistical risk estimates and should not be interpreted as confirmation of fraudulent activity. In a real insurance environment, model outputs should be combined with appropriate investigation procedures, human judgment, governance controls, and applicable regulatory requirements.
