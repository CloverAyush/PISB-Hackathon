import pandas as pd


LOOKUP_PATH = "transaction_lookup.parquet"

REQUIRED_TRANSACTION_COLUMNS = [
    "transaction_id",
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "dest_pagerank",
]


def generate_transaction_id(row_number):
    """
    Match the training pipeline convention:
    f'TX{i:08d}' for i in range(1, len(df) + 1)
    """
    return f"TX{int(row_number):08d}"


def build_transaction_lookup(path=LOOKUP_PATH):
    lookup = pd.read_parquet(path, columns=REQUIRED_TRANSACTION_COLUMNS)

    missing_columns = [
        column
        for column in REQUIRED_TRANSACTION_COLUMNS
        if column not in lookup.columns
    ]
    if missing_columns:
        raise ValueError(
            "Transaction lookup is missing required columns: "
            + ", ".join(missing_columns)
        )

    if lookup["transaction_id"].duplicated().any():
        raise ValueError("Transaction lookup contains duplicate transaction IDs")

    return lookup.set_index("transaction_id", drop=False)


transaction_lookup = build_transaction_lookup()


def lookup_transaction(transaction_id):
    transaction_id = str(transaction_id).strip()

    if transaction_id not in transaction_lookup.index:
        return None

    return transaction_lookup.loc[transaction_id]
