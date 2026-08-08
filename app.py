# ==========================================================
# AI Fraud Investigation System
# Production Deployment Notebook
# ==========================================================

import joblib
import pandas as pd
import numpy as np
import gradio as gr
import shap

import matplotlib.pyplot as plt

from IPython.display import display

# ==========================================================
# Load Production Model
# ==========================================================

print("Loading production assets...")

model = joblib.load("models/fraud_model.joblib")
encoder = joblib.load("models/label_encoder.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

print("✅ Random Forest loaded")
print("✅ Label Encoder loaded")
print("✅ Feature Columns loaded")

print(encoder.classes_)
for i, cls in enumerate(encoder.classes_):
    print(i, "->", cls)

print(f"\nModel expects {len(feature_columns)} features.")

# ==========================================================
# Initialize SHAP Explainer
# ==========================================================

print("Initializing SHAP Explainer...")

explainer = shap.TreeExplainer(model)

print("✅ SHAP Explainer Ready!")

# ==========================================================
# AI Transaction Inference Engine
# ==========================================================

def analyze_transaction(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest
):
    """
    Performs complete transaction analysis.
    Returns everything required by the frontend.
    """

    # -----------------------------
    # Encode transaction type
    # -----------------------------
    transaction_type_encoded = encoder.transform([transaction_type])[0]

    # -----------------------------
    # Feature Engineering
    # Must match training exactly
    # -----------------------------
    errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest

    # -----------------------------
    # Build Input DataFrame
    # -----------------------------
    input_df = pd.DataFrame([{
        "step": step,
        "type": transaction_type_encoded,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "errorBalanceOrig": errorBalanceOrig,
        "errorBalanceDest": errorBalanceDest
    }])

    # Ensure correct feature order
    input_df = input_df[feature_columns]

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    explanation = explainer(input_df)

    # -----------------------------
    # Risk Level
    # -----------------------------
    if probability >= 0.90:
        risk_level = "Critical"
    elif probability >= 0.70:
        risk_level = "High"
    elif probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # -----------------------------
    # Return Analysis Object
    # -----------------------------
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": risk_level,
        "input_df": input_df,
        "shap": explanation
    }

    # ==========================================================
# CELL: Rank SHAP Features
# ==========================================================

import numpy as np

def rank_shap_features(analysis, top_k=3):
    """
    Returns the top SHAP features ranked by absolute contribution.
    """

    shap_values = analysis["shap"].values

    # Handle binary classification SHAP output
    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]
    else:
        shap_values = shap_values[0]

    feature_names = analysis["input_df"].columns.tolist()
    feature_values = analysis["input_df"].iloc[0].to_dict()

    ranked = []

    for feature, shap_value in zip(feature_names, shap_values):

        ranked.append({
            "feature": feature,
            "value": feature_values[feature],
            "shap": float(shap_value),
            "abs_shap": abs(float(shap_value))
        })

    ranked.sort(
        key=lambda x: x["abs_shap"],
        reverse=True
    )

    return ranked[:top_k]

    # ==========================================================
# CELL: Interpret Sender Balance Consistency
# ==========================================================

def interpret_error_balance_orig(feature_info):
    """
    Interpret the engineered sender balance consistency feature.
    """

    value = feature_info["value"]

    observation = {}

    if abs(value) < 1e-6:

        observation["title"] = "Sender Balance Consistency"

        observation["description"] = (
            "The sender's balance is internally consistent with the transaction amount."
        )

    else:

        observation["title"] = "Sender Balance Consistency"

        observation["description"] = (
            f"The sender's balance differs from the expected value by {value:.2f}."
        )

    observation["impact"] = (
        "AI Assessment:This observation increased the model's confidence that the transaction is fraudulent."

        if feature_info["shap"] > 0
        else
        "AI Assessment:This observation increased the model's confidence that the transaction is legitimate."

    )

    observation["importance"] = feature_info["abs_shap"]

    return observation

    # ==========================================================
# CELL: Interpret Receiver Balance Consistency
# ==========================================================

def interpret_error_balance_dest(feature_info):
    """
    Interpret the engineered receiver balance consistency feature.
    """

    value = feature_info["value"]

    observation = {}

    observation["title"] = "Receiver Balance Consistency"

    if abs(value) < 1e-6:

        observation["description"] = (
            "The receiver's balance changed exactly as expected."
        )

    else:

        observation["description"] = (
            f"The receiver's balance differs from the expected value by {value:.2f}."
        )

    observation["impact"] = (
        "Positive contribution to fraud prediction"
        if feature_info["shap"] > 0
        else
        "Negative contribution to fraud prediction"
    )

    observation["importance"] = feature_info["abs_shap"]

    return observation

    # ==========================================================
# CELL: Interpret Transaction Amount
# ==========================================================

def interpret_amount(feature_info):
    """
    Interpret the transaction amount into human-readable categories.
    """

    amount = feature_info["value"]

    observation = {}

    observation["title"] = "Transaction Amount"

    # ---------------------------------------------
    # Categorize transaction value
    # ---------------------------------------------
    if amount < 1000:

        category = "Low-value"

    elif amount < 20000:

        category = "Moderate-value"

    elif amount < 200000:

        category = "High-value"

    else:

        category = "Very high-value"

    observation["description"] = (
        f"This is a {category.lower()} transaction "
        f"({amount:,.2f})."
    )

    # ---------------------------------------------
    # SHAP contribution
    # ---------------------------------------------
    observation["impact"] = (
        "Positive contribution to fraud prediction"
        if feature_info["shap"] > 0
        else
        "Negative contribution to fraud prediction"
    )

    observation["importance"] = feature_info["abs_shap"]

    return observation

    # ==========================================================
# CELL: Interpret Transaction Type
# ==========================================================

def interpret_transaction_type(feature_info):
    """
    Interpret the transaction type.
    """

    transaction_type = feature_info["value"]

    observation = {}

    observation["title"] = "Transaction Type"

    if transaction_type == 1:

        description = "The transaction is a TRANSFER."

    elif transaction_type == 0:

        description = "The transaction is a CASH_OUT."

    else:

        description = f"Unknown transaction type ({transaction_type})."

    observation["description"] = description

    observation["impact"] = (
        "Positive contribution to fraud prediction"
        if feature_info["shap"] > 0
        else
        "Negative contribution to fraud prediction"
    )

    observation["importance"] = feature_info["abs_shap"]

    return observation

    # ==========================================================
# CELL: Explanation Generator
# ==========================================================

def generate_explanation(analysis):
    """
    Generate a human-readable investigation report
    using the top SHAP features.
    """

    # Get the most important features
    ranked = rank_shap_features(analysis, top_k=9)

    observations = []

    # ---------------------------------------------
    # Dispatch to the correct interpreter
    # ---------------------------------------------
    for feature_info in ranked:

        feature = feature_info["feature"]

        if len(observations)==3:
            break

        if feature == "amount":

            observations.append(
                interpret_amount(feature_info)
            )

        elif feature == "type":

            observations.append(
                interpret_transaction_type(feature_info)
            )

        elif feature == "errorBalanceOrig":

            observations.append(
                interpret_error_balance_orig(feature_info)
            )

        elif feature == "errorBalanceDest":

            observations.append(
                interpret_error_balance_dest(feature_info)
            )

    # ---------------------------------------------
    # Build report
    # ---------------------------------------------
    lines = []

    if analysis["prediction"] == 1:
        lines.append("🚨 AI Investigation Summary")
        lines.append("Prediction: FRAUD")
    else:
        lines.append("✅ AI Investigation Summary")
        lines.append("Prediction: LEGITIMATE")

    lines.append("")
    lines.append("Key Findings:")
    lines.append("")

    for obs in observations:

        lines.append(f"• {obs['title']}")
        lines.append(f"  {obs['description']}")
        lines.append(f"  {obs['impact']}")
        lines.append("")

    return "\n".join(lines)

    # ==========================================================
# CELL: Gradio Callback Function
# ==========================================================

import matplotlib.pyplot as plt
import shap

def gradio_predict(
    step,
    transaction_type,
    amount,
    oldbalanceOrg,
    newbalanceOrig,
    oldbalanceDest,
    newbalanceDest
):

    # -------------------------------------------------
    # Run AI inference
    # -------------------------------------------------
    analysis = analyze_transaction(
        step,
        transaction_type,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    )

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    if analysis["prediction"] == 1:
        prediction = "🚨 FRAUD DETECTED"
        action = "Immediately Investigate Transaction"
    else:
        prediction = "✅ LEGITIMATE"
        action = "Approve Transaction"

    probability = f"{analysis['probability']*100:.2f}%"

    # -------------------------------------------------
    # AI Investigation Report
    # -------------------------------------------------
    explanation = generate_explanation(analysis)

    # -------------------------------------------------
    # SHAP Waterfall Plot
    # -------------------------------------------------
    plt.close("all")

    shap.plots.waterfall(
    analysis["shap"][0, :, 1],
    show=False,
    max_display= 8
)

    fig = plt.gcf()
    fig.set_size_inches(10,8)
    fig.subplots_adjust(left= 0.35, right= 0.95)

    # -------------------------------------------------
    # Return to Gradio
    # -------------------------------------------------
    return (
        prediction,
        probability,
        analysis["risk_level"],
        action,
        explanation,
        fig
    )

    # ==========================================================
# CELL: Dashboard Skeleton
# ==========================================================

import gradio as gr

gr.close_all()

with gr.Blocks(
    title="AI Fraud Investigation Platform"
) as demo:

    # ======================================================
    # Header
    # ======================================================
    gr.Markdown("""
# 🛡️ FraudLens
### Explainable Financial Transaction Analysis for Fraud Ananlysts
""")

    with gr.Tabs():

        # ==================================================
        # Transaction Analysis Tab
        # ==================================================
        with gr.Tab("🔍 Transaction Analysis"):

            with gr.Row():

                # ------------------------------------------
                # Left Panel - Inputs
                # ------------------------------------------
                with gr.Column(scale=1):

                    gr.Markdown("## 📋 Transaction Information")

                    step = gr.Number(
                        label="Transaction Time (Simulation Hour)",
                        value=1,
                        precision=0
                    )

                    transaction_type = gr.Dropdown(
                        choices=["TRANSFER", "CASH_OUT"],
                        value="TRANSFER",
                        label="Transaction Type"
                    )

                    amount = gr.Number(
                        label="Transaction Amount",
                        value=1000
                    )

                    oldbalanceOrg = gr.Number(
                        label="Sender Balance (Before)",
                        value=10000
                    )

                    newbalanceOrig = gr.Number(
                        label="Sender Balance (After)",
                        value=9000
                    )

                    oldbalanceDest = gr.Number(
                        label="Receiver Balance (Before)",
                        value=0
                    )

                    newbalanceDest = gr.Number(
                        label="Receiver Balance (After)",
                        value=1000
                    )

                    analyze_btn = gr.Button(
                        "🔍 Analyze Transaction",
                        variant="primary",
                        size="lg"
                    )

                # ------------------------------------------
                # Right Panel - Results
                # ------------------------------------------
                with gr.Column(scale=1):

                    gr.Markdown("## 📊 Analysis Results")

                    prediction_output = gr.Textbox(
                        label="Prediction",
                        interactive=False
                    )

                    probability_output = gr.Textbox(
                        label="Fraud Probability",
                        interactive=False
                    )

                    risk_output = gr.Textbox(
                        label="Risk Level",
                        interactive=False
                    )

                    action_output = gr.Textbox(
                        label="Recommended Action",
                        interactive=False
                    )

                    explanation_output = gr.Textbox(
                        label="AI Investigation Summary",
                        lines=7,
                        interactive=False
                    )

                    shap_output = gr.Plot(
                        label="SHAP Explanation"
                    )

        # ==================================================
        # Model Performance Tab
        # ==================================================
        with gr.Tab("📈 Model Performance"):

            gr.Markdown("## 📈 Production Model Performance")

            with gr.Row():

                with gr.Column():

                    precision_box = gr.Textbox(
                        label="Precision",
                        value="0.999593",
                        interactive=False
                    )

                    recall_box = gr.Textbox(
                        label="Recall",
                        value="0.996347",
                        interactive=False
                    )

                    f1_box = gr.Textbox(
                        label="F1 Score",
                        value="0.997967",
                        interactive=False
                    )

                with gr.Column():

                    roc_box = gr.Textbox(
                        label="ROC-AUC",
                        value="0.998895",
                        interactive=False
                    )

                    pr_box = gr.Textbox(
                        label="PR-AUC",
                        value="0.997170",
                        interactive=False
                    )

                    accuracy_box = gr.Textbox(
                        label="Accuracy",
                        value="99.9988%",
                        interactive=False
                    )

            gr.Markdown("---")

            with gr.Row():

                fp_box = gr.Textbox(
                    label="False Positives",
                    value="1",
                    interactive=False
                )

                fn_box = gr.Textbox(
                    label="False Negatives",
                    value="9",
                    interactive=False
                )

            gr.Markdown("---")

            gr.Markdown("### ⚙️ Model Configuration")

            with gr.Row():

                algorithm_box = gr.Textbox(
                    label="Algorithm",
                    value="Random Forest",
                    interactive=False
                )

                trees_box = gr.Textbox(
                    label="Decision Trees",
                    value="200",
                    interactive=False
                )

                depth_box = gr.Textbox(
                    label="Maximum Depth",
                    value="10",
                    interactive=False
                )

                smote_box = gr.Textbox(
                    label="SMOTE Ratio",
                    value="0.10",
                    interactive=False
                )


    ##############################
    # Event Bindings
    ##############################

    analyze_btn.click(
        fn=gradio_predict,
        inputs=[
            step,
            transaction_type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        ],
        outputs=[
            prediction_output,
            probability_output,
            risk_output,
            action_output,
            explanation_output,
            shap_output
        ]
    )

    demo.launch(
    debug=True,
    share=False,
    server_port=7860
)

