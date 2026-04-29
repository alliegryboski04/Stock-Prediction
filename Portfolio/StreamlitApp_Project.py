import os
import sys
import warnings
import tempfile
import posixpath
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import shap
import __main__

# Setup
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.Custom_Classes import LoanDataCleanerEngineer

# Important for loading the model locally if it was pickled from notebook __main__
__main__.LoanDataCleanerEngineer = LoanDataCleanerEngineer

# File paths
portfolio_dir = os.path.join(project_root, "Portfolio")
xtrain_path = os.path.join(portfolio_dir, "X_train.csv")
model_path = os.path.join(portfolio_dir, "finalized_loan_model.joblib")
explainer_path = os.path.join(portfolio_dir, "explainer_loan.shap")

dataset = pd.read_csv(xtrain_path)
dataset = dataset.loc[:, ~dataset.columns.str.contains("^Unnamed")]

# Secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS session
@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session()
sm_session = sagemaker.Session(boto_session=session)

# App config
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "keys": ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low"]
}

def call_model_api(payload):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    try:
        raw_pred = predictor.predict(payload)

        if isinstance(raw_pred, dict):
            pred_val = raw_pred.get("prediction", [None])[0]
            prob_val = raw_pred.get("probability_default", [None])[0]
        else:
            pred_val = None
            prob_val = None

        mapping = {0: "Fully Paid / Lower Risk", 1: "Charged Off / Higher Risk"}
        return mapping.get(pred_val, str(raw_pred)), prob_val, 200
    except Exception as e:
        return f"Error: {str(e)}", None, 500

def display_explanation(payload):
    if not os.path.exists(model_path):
        st.warning("Local model file not found.")
        return

    if not os.path.exists(explainer_path):
        st.warning("Local SHAP explainer file not found.")
        return

    best_pipeline = joblib.load(model_path)
    explainer = joblib.load(explainer_path)

    input_df = pd.DataFrame([payload])

    # Drop final estimator for preprocessing only
    preprocessing_pipeline = best_pipeline[:-1]
    input_df_transformed = preprocessing_pipeline.transform(input_df)

    try:
        feature_names = best_pipeline[:-1].get_feature_names_out()
        input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    except Exception:
        input_df_transformed = pd.DataFrame(input_df_transformed)

    shap_values = explainer(input_df_transformed)

    st.subheader("SHAP Explanation")
    fig = plt.figure(figsize=(10, 4))

    try:
        if len(shap_values.shape) == 3:
            shap.plots.waterfall(shap_values[0, :, -1], max_display=10, show=False)
            top_feature = pd.Series(
                shap_values[0, :, -1].values,
                index=shap_values[0, :, -1].feature_names
            ).abs().idxmax()
        else:
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            top_feature = pd.Series(
                shap_values[0].values,
                index=shap_values[0].feature_names
            ).abs().idxmax()

        st.pyplot(fig)
        st.info(f"Most influential feature: **{top_feature}**")
    except Exception as e:
        st.warning(f"Could not display SHAP waterfall plot: {e}")

# UI
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Prediction")

with st.form("pred_form"):
    st.subheader("Applicant Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, key in enumerate(MODEL_INFO["keys"]):
        default_val = float(dataset[key].median()) if key in dataset.columns else 0.0

        with cols[i % 2]:
            user_inputs[key] = st.number_input(
                key.replace("_", " ").title(),
                value=default_val
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    base_row = dataset.iloc[0].to_dict()
    base_row.update(user_inputs)

    result, prob_default, status = call_model_api(base_row)

    if status == 200:
        st.metric("Prediction Result", result)
        if prob_default is not None:
            st.metric("Estimated Default Probability", f"{prob_default:.2%}")
        display_explanation(base_row)
    else:
        st.error(result)
