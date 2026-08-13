import numpy as np

from transaction_lookup_layer import transaction_lookup


RECENT_STEP_WINDOW = 24


def _transaction_order(transaction_id):
    return int(str(transaction_id).strip().replace("TX", "", 1))


def _prepare_sender_history():
    history = transaction_lookup[
        ["transaction_id", "nameOrig", "step", "amount"]
    ].copy()
    history["_transaction_order"] = (
        history["transaction_id"]
        .astype(str)
        .str.replace("TX", "", n=1, regex=False)
        .astype(np.int64)
    )
    history = history.sort_values(["nameOrig", "_transaction_order"])
    return history.set_index("nameOrig")


SENDER_HISTORY_BY_ACCOUNT = _prepare_sender_history()


def analyze_sender_behavior(context):
    sender = context["nameOrig"]
    current_step = context["step"]
    current_amount = float(context["amount"])
    current_order = _transaction_order(context["transaction_id"])

    try:
        sender_history = SENDER_HISTORY_BY_ACCOUNT.loc[[sender]]
    except KeyError:
        sender_history = None

    if sender_history is None:
        prior_history = None
    else:
        prior_history = sender_history[
            (sender_history["_transaction_order"] < current_order)
            & (sender_history["step"] < current_step)
        ]

    if prior_history is None or prior_history.empty:
        historical_median = None
        ratio = None
        amount_status = "no_prior_transactions"
        amount_summary = "Sender has no prior transaction amount history."
    else:
        historical_median = float(prior_history["amount"].median())

        if historical_median == 0:
            ratio = None
            amount_status = "zero_historical_median"
            amount_summary = "Sender's historical median transaction amount is zero."
        else:
            ratio = current_amount / historical_median
            amount_status = "available"
            amount_summary = (
                "Transaction amount is "
                f"{ratio:.1f}x the sender's historical median."
            )

    if prior_history is None or prior_history.empty:
        recent_count = 0
    else:
        window_start = current_step - RECENT_STEP_WINDOW
        recent_count = int(
            prior_history[
                (prior_history["step"] >= window_start)
                & (prior_history["step"] < current_step)
            ].shape[0]
        )

    frequency_status = "available"
    frequency_summary = (
        f"Sender made {recent_count} transactions within the recent activity window."
    )

    return {
        "amount_deviation": {
            "current_amount": current_amount,
            "historical_median": historical_median,
            "ratio": ratio,
            "status": amount_status,
        },
        "recent_frequency": {
            "window": RECENT_STEP_WINDOW,
            "transaction_count": recent_count,
            "status": frequency_status,
        },
        "summary": [
            amount_summary,
            frequency_summary,
        ],
    }
