# 🔍 FraudLens AI
### Explainable Financial Fraud Detection using Machine Learning

FraudLens AI is an explainable fraud detection platform designed to identify potentially fraudulent financial transactions and provide human-readable reasoning behind each prediction.

The system uses a **Random Forest classifier** trained on the PaySim financial transaction dataset and combines fraud probability, risk assessment, feature engineering, and **SHAP-based explainability** through an interactive Gradio interface.

> **Important:** PaySim is a synthetic financial transaction dataset. The model demonstrates the feasibility of the proposed approach, but production deployment would require validation and retraining using real-world financial transaction data.

---

## 🚀 Key Features

- 🤖 Fraud detection using Random Forest
- ⚖️ Class-imbalance handling using SMOTE
- 🧩 Transaction-level feature engineering
- 📊 Fraud probability estimation
- 🚦 Risk-level assessment
- 🔎 SHAP-based explainability for individual predictions
- 💬 Human-readable AI investigation summaries
- 🖥️ Interactive Gradio web interface
- 📦 Saved production model and preprocessing artifacts
- 📓 Reproducible notebooks for EDA, experimentation, and development

---

# 🎯 Problem Statement

Financial fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a very small proportion of all transactions.

A useful fraud detection system should therefore do more than simply achieve high accuracy. It should:

1. Identify suspicious transactions.
2. Handle severe class imbalance.
3. Consider meaningful transaction and balance characteristics.
4. Provide interpretable predictions.
5. Support human investigation rather than acting as an unexplained black box.

FraudLens AI addresses these requirements through an end-to-end machine learning pipeline with explainable predictions.

---

# 📊 Dataset

The project uses the **PaySim synthetic mobile-money transaction dataset**.

The original dataset contains approximately **6.3 million transactions** and includes transaction information, account balances, transaction types, identifiers, and fraud labels.

### Target

`isFraud`

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

### Original dataset characteristics

The dataset contains numerical and categorical transaction attributes including:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `nameOrig`
- `nameDest`
- `isFraud`
- `isFlaggedFraud`

---

# 🧹 Data Preparation & Engineering Decisions

The preprocessing pipeline was driven by dataset analysis rather than blindly using every available column.

### Identifier removal

The following columns were removed:

```text
nameOrig
nameDest

These represent account identifiers rather than intrinsic transaction characteristics and were therefore not treated as predictive features.

### PaySim-specific flag removal

`isFlaggedFraud` was removed because it is a simulator-specific flag rather than an intrinsic transaction characteristic. Keeping it could make the model overly dependent on PaySim-specific behavior and reduce the generalizability of the approach.

### Transaction type filtering

EDA showed that fraudulent transactions in the dataset were concentrated in:

- `TRANSFER`
- `CASH_OUT`

Therefore, other transaction types were excluded from the modeling dataset.

The remaining transaction types were encoded numerically for machine learning.

### Target separation

`isFraud` was separated from the input features before model training.

🧩 Feature Engineering

Additional features were created to capture inconsistencies between transaction amounts and account balances.
Sender Balance Error

The expected sender balance relationship is:
oldbalanceOrg - amount ≈ newbalanceOrig

A balance-error feature was created to represent the difference between the expected and observed sender balance.

Receiver Balance Error

The expected receiver balance relationship is:
oldbalanceDest + amount ≈ newbalanceDest

A corresponding receiver balance-error feature was created.

These engineered features allow the model to capture transaction-level balance relationships that may not be captured effectively by the original features independently.

An ablation study was performed to evaluate their actual contribution to model performance.

⚖️ Class Imbalance
Fraudulent transactions represent only a small fraction of the dataset, creating a severe class imbalance problem.

To address this, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data.
The evaluation data was kept untouched.

Original Dataset
       ↓
Train / Test Split
       ↓
SMOTE applied ONLY to training data
       ↓
Random Forest Training
       ↓
Evaluation on untouched test data
This prevents synthetic samples from contaminating the evaluation set.

🌲 Model Selection
The primary classifier is a Random Forest Classifier.

A KNN baseline was also evaluated. Its performance demonstrated the limitations of a distance-based approach for this feature space, particularly after dealing with the highly imbalanced transaction data.

Random Forest was selected because it can model nonlinear relationships and interactions between multiple transaction features without relying on neighborhood similarity.

🔧 Hyperparameter Tuning
Multiple Random Forest configurations were tested.

The number of trees was evaluated using:

50 trees
200 trees
500 trees
A maximum depth of 10 was also compared against deeper trees such as depth 15.
Minimum leaf sample configurations were also tested.

The final configuration selected from these experiments was:
Algorithm       : Random Forest Classifier
Number of Trees : 200
Maximum Depth   : 10
Random State    : 42
n_jobs          : -1

The selected configuration provided the best balance among the tested configurations without unnecessarily increasing model complexity.

📊 Model Evaluation
Because this is a highly imbalanced fraud-detection problem, accuracy alone is not sufficient.

The evaluation includes:
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
Confusion Matrix
False Positives
False Negatives

Particular attention is given to precision and recall, since both missed fraud and incorrectly flagged legitimate transactions have practical consequences.

🔎 Explainable AI with SHAP
Fraud detection predictions should not be treated as unexplained black-box decisions.

FraudLens AI uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

For each transaction, SHAP identifies which features contributed toward or away from the fraud prediction.

Positive SHAP contribution
        ↓
Pushes prediction toward fraud

Negative SHAP contribution
        ↓
Pushes prediction away from fraud

The application presents these contributions through a SHAP visualization and a human-readable AI investigation summary.

🖥️ Application
The trained model and preprocessing artifacts are integrated into an interactive Gradio application.

The user provides transaction information including:

Transaction type
Transaction amount
Sender balance before transaction
Sender balance after transaction
Receiver balance before transaction
Receiver balance after transaction

The application produces:
Fraud Prediction
FRAUD
or
LEGITIMATE

Fraud Probability
The model's estimated probability of fraud.

Risk Assessment
The prediction is converted into a risk level for easier interpretation.

AI Investigation Summary
The application identifies important model contributions and converts them into human-readable observations.

SHAP Explanation
A transaction-level SHAP visualization shows the features that had the strongest influence on the prediction.

🏗️ Project Structure
FraudLens-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── fraud_model.joblib
│   ├── label_encoder.joblib
│   └── feature_columns.joblib
│
└── notebooks/
    ├── EDA_Fraud_Detection.ipynb
    ├── Training.ipynb
    ├── KNN_Test.ipynb
    └── Gradio_development.ipynb


🔄 End-to-End Workflow
PaySim Dataset
      ↓
EDA & Data Analysis
      ↓
Data Cleaning
      ↓
Remove Identifiers & PaySim-specific Flag
      ↓
Transaction Type Filtering
      ↓
Feature Engineering
      ↓
Train / Test Split
      ↓
SMOTE on Training Data
      ↓
Random Forest Training
      ↓
Model Evaluation
      ↓
Model Serialization
      ↓
Gradio Application
      ↓
Fraud Prediction + Risk Assessment
      ↓
SHAP Explanation
      ↓
AI Investigation Summary

🧪 Experiments & Validation
The project included multiple experiments rather than relying on a single model configuration.

KNN Comparison
A KNN baseline was evaluated to establish a comparison against the tree-based approach.

Random Forest Tuning
The number of estimators, tree depth, and minimum leaf samples were experimentally evaluated.

Ablation Study
The engineered balance-error features were removed and the model was reevaluated.
This demonstrated their contribution to the model's performance on the PaySim dataset rather than assuming that feature engineering was beneficial.


⚠️ Limitations & Feasibility

The primary limitation of this project is the dataset.

PaySim is a synthetic financial transaction simulator, so patterns learned from it cannot automatically be assumed to represent real banking transactions.

In particular, the engineered balance-consistency features showed strong predictive value on PaySim. Their effectiveness should therefore be independently validated using real-world financial transaction data before production deployment.

The current system demonstrates the technical feasibility of an explainable fraud-detection pipeline, rather than claiming that the trained model itself is production-ready for real banking systems.

A real deployment would require:

Validation on real financial data
Dataset-specific feature engineering
Threshold calibration
Model monitoring
Data-drift detection
Security and privacy controls
Integration with existing financial systems

🔮 Future Improvements

Validation using real-world financial datasets
Threshold optimization based on business costs
REST API deployment
Real-time transaction processing
Model monitoring and drift detection
Docker-based deployment
Cloud deployment
Continuous model retraining

🛠️ Technology Stack

Python
Pandas
NumPy
Scikit-learn
imbalanced-learn
SHAP
Gradio
Joblib
Matplotlib

⚙️ Installation
Clone the repository:

git clone https://github.com/CloverAyush/PISB-Hackathon
cd FraudLens-AI

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

▶️ Running the Application

After activating the virtual environment:

python app.py

The Gradio interface will launch locally.

📌 Project Summary

FraudLens AI combines:

EDA → Feature Engineering → SMOTE → Random Forest → Risk Assessment → SHAP → Human-readable Investigation
The objective is not simply to maximize classification accuracy, but to demonstrate an interpretable, testable, and practically feasible fraud-detection workflow.