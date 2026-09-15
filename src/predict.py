import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "customer_churn_logistic_regression.pkl")

FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")


model = joblib.load(MODEL_PATH)
model_features = joblib.load(FEATURES_PATH)


def predict_churn(customer_data):

    input_df = pd.DataFrame([customer_data])

    input_encoded = pd.get_dummies(input_df)

    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    input_encoded = input_encoded.astype(int)

    probability = model.predict_proba(input_encoded)[0][1]

    prediction = model.predict(input_encoded)[0]

    if probability >= 0.70:
        risk = "HIGH RISK"

    elif probability >= 0.40:
        risk = "MEDIUM RISK"

    else:
        risk = "LOW RISK"

    return probability, prediction, risk
