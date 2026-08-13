# ==========================================================
# AI Fraud Investigation System
# Production Deployment Notebook
# ==========================================================

import joblib
import pandas as pd
import numpy as np
import gradio as gr
import shap
import pickle
import igraph as ig

import matplotlib.pyplot as plt

from IPython.display import display

from transaction_lookup_layer import lookup_transaction, transaction_lookup
from sender_behavior import analyze_sender_behavior

# ==========================================================
# Load Production Model
# ==========================================================

print("Loading production assets...")

model = joblib.load("models_all/fraud_model_g.joblib")
encoder = joblib.load("models_all/label_encoder_g.joblib")
feature_columns = joblib.load("models_all/feature_column_g.joblib")

print("✅ Random Forest loaded")
print("✅ Label Encoder loaded")
print("✅ Feature Columns loaded")

print(f"\nModel expects {len(feature_columns)} features.")

df_lookup = transaction_lookup

print("Lookup loaded:", df_lookup.shape)
print(df_lookup.head())
print(df_lookup.columns.tolist())

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
    newbalanceDest,
    raw_transaction=None
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
    errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest
    dest_pagerank = 0.0

    if raw_transaction is not None:
        dest_pagerank = raw_transaction["dest_pagerank"]

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
        "errorBalanceDest": errorBalanceDest,
        "dest_pagerank": dest_pagerank
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
        "shap": explanation,
        "raw_transaction": raw_transaction
    }


def analyze_transaction_from_lookup(transaction_id):
    raw_transaction = lookup_transaction(transaction_id)

    if raw_transaction is None:
        return None

    return analyze_transaction(
        raw_transaction["step"],
        raw_transaction["type"],
        raw_transaction["amount"],
        raw_transaction["oldbalanceOrg"],
        raw_transaction["newbalanceOrig"],
        raw_transaction["oldbalanceDest"],
        raw_transaction["newbalanceDest"],
        raw_transaction=raw_transaction
    )

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
            "abs_shap": abs(float(shap_value)),
            "raw_transaction": analysis.get("raw_transaction")
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

    raw_transaction = feature_info.get("raw_transaction")
    transaction_type = (
        raw_transaction["type"]
        if raw_transaction is not None
        else feature_info["value"]
    )

    observation = {}

    observation["title"] = "Transaction Type"

    if transaction_type == "TRANSFER" or transaction_type == 1:

        description = "The transaction is a TRANSFER."

    elif transaction_type == "CASH_OUT" or transaction_type == 0:

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

def _format_money(value):
    if value is None:
        return "Insufficient historical data"

    return f"{float(value):,.2f}"


def _format_ratio(value):
    if value is None:
        return "Insufficient historical data"

    return f"{float(value):.2f}x"


def _format_amount_deviation_card(behavior):
    amount_deviation = behavior["amount_deviation"]
    status = amount_deviation["status"]
    status_text = (
        "Insufficient historical data"
        if status in {"no_prior_transactions", "zero_historical_median"}
        else status.replace("_", " ").title()
    )

    return f"""
### Amount Deviation

**Current transaction amount:** {_format_money(amount_deviation["current_amount"])}

**Historical median amount:** {_format_money(amount_deviation["historical_median"])}

**Deviation ratio:** {_format_ratio(amount_deviation["ratio"])}

**Status:** {status_text}
"""


def _format_recent_frequency_card(behavior):
    recent_frequency = behavior["recent_frequency"]

    return f"""
### Recent Transaction Frequency

**Recent activity window:** {recent_frequency["window"]} steps

**Transactions in window:** {recent_frequency["transaction_count"]}

**Status:** {recent_frequency["status"].replace("_", " ").title()}
"""


def _format_behavior_summary(behavior):
    return "\n".join(
        f"- {line}"
        for line in behavior["summary"]
    )


def _empty_plot():
    plt.close("all")
    return None


def gradio_predict(transaction_id):

    # -------------------------------------------------
    # Lookup transaction and run AI inference
    # -------------------------------------------------
    raw_transaction = lookup_transaction(transaction_id)

    if raw_transaction is None:
        message = "Transaction ID not found. Please enter a valid transaction ID."
        return (
            message,
            "",
            "",
            "",
            message,
            _empty_plot(),
            "",
            "",
            "",
        )

    try:
        analysis = analyze_transaction(
            raw_transaction["step"],
            raw_transaction["type"],
            raw_transaction["amount"],
            raw_transaction["oldbalanceOrg"],
            raw_transaction["newbalanceOrig"],
            raw_transaction["oldbalanceDest"],
            raw_transaction["newbalanceDest"],
            raw_transaction=raw_transaction
        )
    except Exception as exc:
        message = f"Unable to analyze this transaction: {exc}"
        return (
            message,
            "",
            "",
            "",
            message,
            _empty_plot(),
            "",
            "",
            "",
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

    behavior = analyze_sender_behavior(raw_transaction)

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
        fig,
        _format_amount_deviation_card(behavior),
        _format_recent_frequency_card(behavior),
        _format_behavior_summary(behavior)
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

                    gr.Markdown("## Transaction Investigation")

                    transaction_id_input = gr.Textbox(
                        label="Transaction ID",
                        placeholder="TX00000001"
                    )

                    analyze_btn = gr.Button(
                        "Analyze Transaction",
                        variant="primary",
                        size="lg"
                    )

                # ------------------------------------------
                # Right Panel - Results
                # ------------------------------------------
                with gr.Column(scale=1):

                    gr.Markdown("## Risk Assessment")

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

                    gr.Markdown("## Why was this transaction flagged?")

                    explanation_output = gr.Textbox(
                        label="AI Investigation Summary",
                        lines=7,
                        interactive=False
                    )

                    gr.Markdown("## SHAP Explanation")

                    shap_output = gr.Plot(
                        label="SHAP Explanation"
                    )

            gr.Markdown("---")
            gr.Markdown("## Behavioural Analysis")

            with gr.Row():

                with gr.Column():

                    amount_deviation_output = gr.Markdown()

                with gr.Column():

                    recent_frequency_output = gr.Markdown()

            behavior_summary_output = gr.Markdown()

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
            transaction_id_input
        ],
        outputs=[
            prediction_output,
            probability_output,
            risk_output,
            action_output,
            explanation_output,
            shap_output,
            amount_deviation_output,
            recent_frequency_output,
            behavior_summary_output
        ]
    )

    demo.launch(
    debug=True,
    share=True,
    server_port=7860
)
