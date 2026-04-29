import pandas as pd
import numpy as np
import json


TOP_INPUT_KEYS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "issue_d",
    "earliest_cr_line"
]


def load_reference_row(csv_path: str) -> pd.Series:
    """
    Load X_train.csv and return the first row as a baseline template.
    """
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df.iloc[0].copy()


def build_loan_input(user_inputs: dict, csv_path: str) -> pd.DataFrame:
    """
    Create a one-row DataFrame for prediction by starting from a reference row
    and replacing only the user-provided values.
    """
    row = load_reference_row(csv_path)

    for key, value in user_inputs.items():
        row[key] = value

    return pd.DataFrame([row])


def parse_json_payload(request_body):
    """
    Safely parse JSON request bodies from local app usage.
    """
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    return json.loads(request_body)
