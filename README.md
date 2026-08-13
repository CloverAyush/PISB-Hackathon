# FraudLens AI

Explainable financial fraud detection using machine learning, graph-derived context, and transaction-level investigation summaries.

FraudLens AI is a fraud-detection prototype built on the PaySim financial transaction dataset. The system predicts whether a transaction is likely to be fraudulent and explains the prediction in a way that supports human investigation.

The final application uses a Random Forest classifier, engineered balance-consistency features, destination PageRank from an account transaction graph, SHAP explanations, and a Gradio interface for transaction lookup and analysis.

> Important: PaySim is a synthetic financial transaction dataset. This project demonstrates an explainable fraud-detection workflow, but the trained model should not be treated as production-ready for real banking data without retraining, validation, threshold calibration, and monitoring.

---

## Key Features

- Fraud detection using a Random Forest classifier
- Severe class-imbalance handling with SMOTE applied only to training data
- Transaction-level balance-error feature engineering
- Destination PageRank as the retained graph-derived structural feature
- Fraud probability and risk-level assessment
- SHAP-based explanations for individual predictions
- Human-readable investigation summaries
- Transaction ID lookup for demo and testing workflows
- Behavioural/contextual proxy analysis based on available PaySim transaction history
- Interactive Gradio web interface
- Saved model and preprocessing artifacts
- Reproducible notebooks for EDA, graph experimentation, training, and application development

---

## Problem Statement

Financial fraud detection is a highly imbalanced classification problem. Fraudulent transactions represent only a small fraction of all transactions, so a model can appear accurate while still missing the cases that matter most.

A useful fraud-detection system should:

1. Identify suspicious transactions.
2. Handle severe class imbalance.
3. Use meaningful transaction and account-state features.
4. Avoid data leakage.
5. Provide interpretable predictions.
6. Support human investigation rather than acting as an unexplained black box.

FraudLens AI addresses these goals through an end-to-end pipeline:

```text
EDA -> Feature Engineering -> Graph Analysis -> SMOTE -> Random Forest -> SHAP -> Investigation UI
```

---

## Dataset

The project uses the PaySim synthetic mobile-money transaction dataset.

The original dataset contains approximately 6.3 million financial transactions and 11 original columns, including transaction attributes, account balances, sender and receiver identifiers, simulator flags, and fraud labels.

### Target Variable

```text
isFraud
0 = Legitimate transaction
1 = Fraudulent transaction
```

### Original Columns

- `step`
- `type`
- `amount`
- `nameOrig`
- `oldbalanceOrg`
- `newbalanceOrig`
- `nameDest`
- `oldbalanceDest`
- `newbalanceDest`
- `isFraud`
- `isFlaggedFraud`

---

## EDA Findings and Engineering Decisions

The preprocessing and modelling pipeline was driven by the EDA rather than by blindly using every available column.

### 1. Data Quality

The dataset contains no missing values, so no imputation or row removal was required. The main preparation work focused on feature selection, leakage prevention, imbalance handling, and feature engineering.

### 2. Identifier Handling

`nameOrig` and `nameDest` are high-cardinality account identifiers. They were not used directly as classifier inputs because the raw ID values do not represent reusable predictive categories.

However, the identifiers were not discarded completely. They were used later to construct account-level graph features.

Decision:

- Exclude `nameOrig` and `nameDest` as direct model features.
- Preserve them for graph-based analysis and transaction lookup.

### 3. PaySim-Specific Flag

`isFlaggedFraud` was removed because it is a PaySim simulator flag rather than an intrinsic transaction characteristic. Keeping it could make the model depend on simulator-specific behaviour rather than learning from transaction and account-state patterns.

Decision:

- Remove `isFlaggedFraud` from model inputs.
- Use `isFraud` only as the supervised learning target.

### 4. Class Imbalance

Fraud is extremely rare in the dataset. Because of this, accuracy alone is not an appropriate primary metric.

Decision:

- Split the data before oversampling.
- Apply SMOTE only to the training data.
- Keep validation/test data untouched.
- Evaluate using Precision, Recall, F1 Score, ROC-AUC, PR-AUC, confusion matrix, false positives, and false negatives.

```text
Original Dataset
      |
Train / Test Split
      |
SMOTE on Training Data Only
      |
Random Forest Training
      |
Evaluation on Untouched Test Data
```

### 5. Transaction Type Scope

EDA showed that fraud in PaySim occurs only in:

- `TRANSFER`
- `CASH_OUT`

`PAYMENT`, `DEBIT`, and `CASH_IN` contain no fraudulent examples in this dataset.

Decision:

- Restrict the supervised modelling dataset to `TRANSFER` and `CASH_OUT`.
- Treat this as a PaySim-specific modelling decision, not a universal claim about real-world fraud.
- Encode the remaining transaction types for machine learning.

### 6. Transaction Amount

Transaction amount contains useful signal, but it is not sufficient by itself. Legitimate and fraudulent transaction amounts overlap, and legitimate transactions also include high-value outliers.

Decision:

- Retain `amount`.
- Do not treat amount as a standalone fraud rule.
- Use a model that can combine amount with balance, type, engineered, and graph-derived features.

### 7. Balance-Error Feature Engineering

A transaction changes account state. The EDA showed that balance relationships are important, so two balance-consistency features were engineered.

Sender balance expectation:

```text
oldbalanceOrg - amount ~= newbalanceOrig
```

Receiver balance expectation:

```text
oldbalanceDest + amount ~= newbalanceDest
```

Engineered features:

- `errorBalanceOrig`
- `errorBalanceDest`

These features represent deviations between expected and observed balance transitions.

Decision:

- Retain both balance-error features.
- Interpret them as transaction-state consistency signals.
- Avoid treating them as standalone fraud rules because some balance behaviour is PaySim-specific.

---

## Graph-Based Feature Scope

The account identifiers create a natural directed transaction network:

```text
sender account -> receiver account
```

In this graph:

- Accounts are nodes.
- Transactions are directed edges.
- `nameOrig` identifies the sender node.
- `nameDest` identifies the receiver node.

Graph-derived features were empirically tested to determine whether account-network structure added useful signal beyond raw transaction and balance features.

### Degree and Flow Signals

Degree-style and flow-style graph features were investigated, including origin/destination degree patterns and related account-flow signals.

The EDA showed some structural differences, especially around destination-side connectivity. However, most degree/flow-style graph signals were weak, limited, or not strong enough to emphasize in the final implemented classifier.

Decision:

- Use degree and flow analysis as supporting EDA evidence.
- Do not overstate these features as strong final predictors.
- Retain only graph-derived features that added useful empirical signal.

### Destination PageRank

Destination PageRank provided a more useful structural signal. It describes the receiver account's position in the transaction network rather than simply counting direct connections.

In the final system, `dest_pagerank` is mapped onto each transaction using the receiver account.

Decision:

- Retain `dest_pagerank` as the final graph-derived model feature.
- Compute graph features at account level and map them back to transactions.
- Use PageRank as network context, not as a standalone fraud rule.

---

## Behavioural and Contextual Scope

The final application includes a behavioural analysis section, but this is intentionally scoped to the information PaySim actually provides.

PaySim does not provide rich real-world customer history such as:

- verified account age
- device history
- login behaviour
- merchant history
- identity verification status
- known prior fraud history
- real customer profiles

Because of this, FraudLens AI does not claim to perform full historical customer profiling.

Instead, the behavioural section uses transaction-context proxies available from the lookup data:

- current transaction amount
- sender's previous transaction amounts when available
- amount deviation from the sender's historical median when available
- recent sender transaction frequency within a step-based window
- transaction type and account-state context

Decision:

- Frame behavioural output as contextual proxy analysis.
- Show "insufficient historical data" when prior sender history is unavailable.
- Avoid unsupported claims about long-term real-world sender behaviour.

---

## Final Model Feature Set

The final classifier uses 10 features:

### Original Transaction and Account-State Features

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

### Engineered Features

- `errorBalanceOrig`
- `errorBalanceDest`

### Graph-Derived Feature

- `dest_pagerank`

---

## Model Selection

The primary model is a Random Forest classifier.

A KNN baseline was also evaluated. Its performance showed the limitations of a distance-based approach for this feature space, especially under severe class imbalance.

Random Forest was selected because it can model nonlinear relationships and interactions between:

- transaction amount
- transaction type
- sender and receiver balance states
- engineered balance-error features
- destination PageRank

---

## Hyperparameter Tuning

Multiple Random Forest configurations were tested.

The number of trees was evaluated using:

- 50 trees
- 200 trees
- 500 trees

Tree depth and minimum leaf sample configurations were also compared.

Final selected configuration:

```text
Algorithm       : Random Forest Classifier
Number of Trees : 200
Maximum Depth   : 10
Random State    : 42
n_jobs          : -1
```

The selected configuration provided a strong balance between performance and model complexity among the tested configurations.

---

## Model Evaluation

Because this is a highly imbalanced fraud-detection task, evaluation focuses on more than accuracy.

The application reports the final model performance as:

```text
Precision : 0.999593
Recall    : 0.996347
F1 Score  : 0.997967
ROC-AUC   : 0.998895
PR-AUC    : 0.997170
Accuracy  : 99.9988%
FP        : 1
FN        : 9
```

Precision and recall are especially important because both false alarms and missed fraud cases have practical consequences.

---

## Explainable AI with SHAP

Fraud detection predictions should not be treated as unexplained black-box decisions.

FraudLens AI uses SHAP to explain individual predictions. For each analyzed transaction, SHAP identifies which features pushed the prediction toward fraud or toward legitimacy.

The application presents:

- fraud probability
- risk level
- recommended action
- SHAP waterfall plot
- human-readable investigation summary

The explanation layer focuses on features that can be communicated clearly to an analyst, such as transaction amount, transaction type, balance consistency, and model contribution direction.

---

## Application Workflow

The Gradio application is centered on transaction investigation.

The user enters a transaction ID. The app then:

1. Looks up the original transaction from the local lookup table.
2. Reconstructs the model input features.
3. Adds engineered balance-error features.
4. Adds the retained destination PageRank feature.
5. Runs the Random Forest prediction.
6. Computes the fraud probability and risk level.
7. Generates SHAP explanations.
8. Displays behavioural/contextual proxy analysis.
9. Provides a human-readable investigation summary.

Outputs include:

- Fraud or legitimate prediction
- Fraud probability
- Risk level
- Recommended action
- SHAP explanation
- Amount deviation card
- Recent sender-frequency card
- Behavioural/contextual summary

---

## Project Structure

```text
PISB TechRush/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- transaction_lookup_layer.py
|-- sender_behavior.py
|-- transaction_lookup.parquet
|-- DEMO_IDS.md
|
|-- models_all/
|   |-- fraud_model_g.joblib
|   |-- label_encoder_g.joblib
|   |-- feature_column_g.joblib
|
|-- ID_CSV/
|   |-- test_fraud_ids_supported.csv
|   |-- test_legit_ids_supported.csv
|   |-- train_fraud_ids_supported.csv
|   |-- train_legit_ids_supported.csv
|
|-- notebooks/
|   |-- EDA_Fraud_Detection.ipynb
|   |-- Training.ipynb
|   |-- KNN_Test.ipynb
|   |-- Gradio_development.ipynb
|   |-- Implemening_Graph_Features.ipynb
|   |-- Gradio_development_gpt.ipynb
|
|-- tests/
|   |-- test_sender_behavior.py
|   |-- test_transaction_lookup_layer.py
```

`transaction_lookup.parquet` is a large local artifact used by the demo lookup layer. It should remain local and does not need to be pushed to GitHub.

---

## Experiments and Validation

The project included several experiments rather than relying on a single modelling attempt.

### KNN Baseline

A KNN baseline was evaluated to compare against the tree-based approach.

### Random Forest Tuning

Random Forest configurations were tested across tree count, depth, and minimum leaf settings.

### Balance-Feature Ablation

The engineered balance-error features were removed and the model was reevaluated. This tested whether the features contributed empirically instead of assuming that they were useful.

### Graph Feature Testing

Graph-derived account features were tested empirically. Most degree/flow-style graph signals were weak or limited, while destination PageRank provided useful structural context and was retained.

### Lookup and Behaviour Tests

The transaction lookup layer and sender-behaviour helper are covered by focused tests to support reliable demo behaviour.

---

## Limitations

The main limitation of this project is the dataset.

PaySim is synthetic, so learned patterns cannot automatically be assumed to represent real banking behaviour. In particular:

- transaction type filtering is PaySim-specific
- balance-error features may reflect simulator behaviour
- graph structure comes from simulated account relationships
- behavioural analysis is limited to transaction-context proxies
- sender history is only as rich as the available PaySim lookup data

The current system demonstrates the feasibility of an explainable fraud-detection workflow. A real deployment would require:

- validation on real financial transaction data
- dataset-specific feature engineering
- business-driven threshold calibration
- model monitoring
- data-drift detection
- security and privacy controls
- integration with existing financial systems

---

## Future Improvements

- Validate on real-world financial datasets
- Improve threshold selection using business cost assumptions
- Add richer customer-history features where legally and ethically available
- Add model monitoring and drift detection
- Add REST API support
- Support real-time transaction processing
- Package with Docker
- Add cloud deployment once local demo assets are stable
- Add continuous retraining workflows

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- imbalanced-learn
- SHAP
- Gradio
- Joblib
- igraph
- Matplotlib
- Seaborn
- PyArrow
- Requests
- IPython

---

## Installation

Clone the repository:

```bash
git clone https://github.com/CloverAyush/PISB-Hackathon
cd "PISB TechRush"
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

After activating the virtual environment, run:

```bash
python app.py
```

The Gradio interface launches locally. Use a supported transaction ID from `DEMO_IDS.md` or the files in `ID_CSV/`.

---

## Project Summary

FraudLens AI is not only a fraud classifier. It is an explainable transaction-investigation prototype.

The EDA showed that fraud in PaySim is highly imbalanced, concentrated in `TRANSFER` and `CASH_OUT`, not separable by transaction amount alone, and better represented through a combination of transaction attributes, balance-consistency features, and selected network context.

The final system reflects those findings through:

```text
Transaction Lookup
      |
Feature Reconstruction
      |
Balance-Error Engineering
      |
Destination PageRank Context
      |
Random Forest Prediction
      |
SHAP Explanation
      |
Behavioural/Contextual Proxy Summary
      |
Analyst-Facing Investigation Output
```

The objective is to demonstrate an interpretable, testable, and factually scoped fraud-detection workflow that supports analyst decision-making.
