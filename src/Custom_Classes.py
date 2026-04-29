
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LoanDataCleanerEngineer(BaseEstimator, TransformerMixin):
    """
    Deterministic cleaning + feature engineering for LendingClub default modeling.
    Designed so the same logic can be reused in EDA, model training, tuning, and deployment.
    """
    def __init__(self):
        self.percent_cols = ["int_rate", "revol_util"]
        self.numeric_like_cols = [
            "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment",
            "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
            "open_acc", "pub_rec", "revol_bal", "total_acc",
            "pub_rec_bankruptcies", "fico_range_low", "fico_range_high"
        ]
        self.text_cols = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state", "term"]

    def fit(self, X, y=None):
        return self

    def _clean_percentage(self, s):
        return pd.to_numeric(
            s.astype(str).str.replace("%", "", regex=False).replace(["nan", "None", ""], np.nan),
            errors="coerce"
        )

    def _clean_term(self, s):
        return pd.to_numeric(s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")

    def _clean_emp_length(self, s):
        cleaned = (
            s.astype(str)
             .str.lower()
             .str.replace(r"\+ years", "", regex=True)
             .str.replace(r"years?", "", regex=True)
             .str.replace(r"<\s*1", "0", regex=True)
             .str.replace(r"[^0-9]", "", regex=True)
             .replace("", np.nan)
        )
        return pd.to_numeric(cleaned, errors="coerce")

    def transform(self, X):
        X = X.copy()

        # 1) Clean percentage columns
        for col in self.percent_cols:
            if col in X.columns:
                X[col] = self._clean_percentage(X[col])

        # 2) Clean term
        if "term" in X.columns:
            X["term"] = self._clean_term(X["term"])

        # 3) Clean employment length
        if "emp_length" in X.columns:
            X["emp_length"] = self._clean_emp_length(X["emp_length"])

        # 4) Parse dates
        for col in ["issue_d", "earliest_cr_line"]:
            if col in X.columns:
                X[col] = pd.to_datetime(X[col], format="%b-%Y", errors="coerce")

        # 5) Standardize categorical text
        for col in [c for c in self.text_cols if c in X.columns]:
            X[col] = X[col].astype(str).str.strip().str.upper().replace({"NAN": np.nan, "NONE": np.nan})

        # 6) Coerce numeric-like columns
        for col in [c for c in self.numeric_like_cols if c in X.columns]:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # -------- Feature engineering --------
        if {"fico_range_low", "fico_range_high"}.issubset(X.columns):
            X["fico_avg"] = (X["fico_range_low"] + X["fico_range_high"]) / 2

        if {"annual_inc", "loan_amnt"}.issubset(X.columns):
            X["income_to_loan_ratio"] = X["annual_inc"] / X["loan_amnt"].replace(0, np.nan)

        if {"installment", "annual_inc"}.issubset(X.columns):
            X["installment_to_income_ratio"] = (12 * X["installment"]) / X["annual_inc"].replace(0, np.nan)

        if {"revol_bal", "annual_inc"}.issubset(X.columns):
            X["revol_bal_to_income_ratio"] = X["revol_bal"] / X["annual_inc"].replace(0, np.nan)

        if {"issue_d", "earliest_cr_line"}.issubset(X.columns):
            X["credit_history_months"] = (X["issue_d"] - X["earliest_cr_line"]).dt.days / 30.44

        if {"open_acc", "total_acc"}.issubset(X.columns):
            X["open_acc_to_total_acc_ratio"] = X["open_acc"] / X["total_acc"].replace(0, np.nan)

        if {"inq_last_6mths", "credit_history_months"}.issubset(X.columns):
            X["inq_per_credit_year"] = 12 * X["inq_last_6mths"] / X["credit_history_months"].replace(0, np.nan)

        if {"delinq_2yrs", "total_acc"}.issubset(X.columns):
            X["delinq_to_total_acc_ratio"] = X["delinq_2yrs"] / X["total_acc"].replace(0, np.nan)

        if "pub_rec_bankruptcies" in X.columns:
            X["pubrec_bankruptcies_flag"] = (X["pub_rec_bankruptcies"].fillna(0) > 0).astype(int)

        if "verification_status" in X.columns:
            X["verified_income_flag"] = X["verification_status"].isin(["VERIFIED", "SOURCE VERIFIED"]).astype(int)

        if "home_ownership" in X.columns:
            X["mortgage_or_own_flag"] = X["home_ownership"].isin(["MORTGAGE", "OWN"]).astype(int)

        if "annual_inc" in X.columns:
            X["log_annual_inc"] = np.log1p(X["annual_inc"].clip(lower=0))

        if "revol_bal" in X.columns:
            X["log_revol_bal"] = np.log1p(X["revol_bal"].clip(lower=0))

        # 7) Replace inf values
        X = X.replace([np.inf, -np.inf], np.nan)

        # 8) Drop raw dates after engineering
        X = X.drop(columns=[c for c in ["issue_d", "earliest_cr_line"] if c in X.columns], errors="ignore")

        return X
