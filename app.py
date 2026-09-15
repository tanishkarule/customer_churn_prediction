# To run program run the command streamlit run app.py.
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")


# --------------------------------------------------
# Load Model
# --------------------------------------------------


@st.cache_resource
def load_model():

    model = joblib.load("models/customer_churn_logistic_regression.pkl")

    features = joblib.load("models/model_features.pkl")

    return model, features


model, model_features = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📊 Customer Churn Prediction System")

st.write(
    "Predict whether a customer is likely to churn " "using a Machine Learning model."
)

st.divider()


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox("Gender", ["Male", "Female"])

    senior_citizen = st.selectbox("Senior Citizen", [0, 1])

    partner = st.selectbox("Partner", ["Yes", "No"])

    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)


with col2:

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])


with col3:

    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )


# --------------------------------------------------
# Account Information
# --------------------------------------------------

st.subheader("Account Information")

col1, col2, col3 = st.columns(3)


with col1:

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])


with col2:

    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])


with col3:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )


# --------------------------------------------------
# Charges
# --------------------------------------------------

st.subheader("Charges")

col1, col2 = st.columns(2)


with col1:

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)


with col2:

    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=monthly_charges * max(tenure, 1)
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button("🔍 Predict Customer Churn", use_container_width=True):

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([customer_data])

    # One-hot encoding
    input_encoded = pd.get_dummies(input_df)

    # Match training features
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    input_encoded = input_encoded.astype(int)

    # Prediction
    probability = model.predict_proba(input_encoded)[0][1]

    prediction = model.predict(input_encoded)[0]

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric("Churn Probability", f"{probability:.2%}")

    with col2:

        if prediction == 1:

            st.error("⚠️ HIGH CHURN POSSIBILITY")

        else:

            st.success("✅ CUSTOMER LIKELY TO STAY")

    with col3:

        if probability >= 0.70:

            risk = "High"

        elif probability >= 0.40:

            risk = "Medium"

        else:

            risk = "Low"

        st.metric("Risk Level", risk)

    st.progress(float(probability))

    st.info(
        "This prediction is generated by a machine-learning "
        "model trained on historical customer data."
    )
