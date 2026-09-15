# 📊 Customer Churn Prediction

> A complete Machine Learning project that predicts whether a telecom customer is likely to churn, compares multiple classification algorithms, handles class imbalance using SMOTE, and provides an interactive Streamlit web application for real-time predictions.

---

## 🚀 Project Overview

Customer churn is one of the major challenges faced by subscription-based businesses.

When customers leave a service, companies lose recurring revenue and may also spend significant resources acquiring new customers. Predicting customers who are likely to churn can help businesses take preventive retention actions.

This project uses historical telecom customer data to build a machine learning classification system that predicts whether a customer is likely to leave the service.

The project covers the complete machine learning workflow:

**Data Collection → Data Cleaning → EDA → Feature Engineering → SMOTE → Model Training → Model Comparison → Evaluation → Model Saving → Streamlit Deployment**

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze customer churn patterns
* Identify customer characteristics associated with churn
* Clean and preprocess the dataset
* Handle categorical variables using One-Hot Encoding
* Handle class imbalance using SMOTE
* Train multiple classification models
* Compare model performance using multiple evaluation metrics
* Select the best-performing model
* Build an interactive web application for churn prediction
* Make the project reproducible and GitHub-ready

---

## 🧠 Problem Statement

Given information about a telecom customer, predict whether the customer is likely to:

* **Stay with the service**
* **Churn from the service**

### Target Variable

The target variable is:

```text
Churn
```

It is converted into:

```text
0 → Customer stayed
1 → Customer churned
```

---

## 📂 Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains customer information related to:

* Demographics
* Customer tenure
* Phone services
* Internet services
* Online security
* Online backup
* Device protection
* Technical support
* Streaming services
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges
* Customer churn status

### Dataset File

```text
Data/
└── Telco-Customer-Churn.csv
```

---

## 🛠️ Technologies & Libraries

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Imbalanced Data

* Imbalanced-learn
* SMOTE

### Model Persistence

* Joblib

### Application

* Streamlit

### Development

* Jupyter Notebook
* Visual Studio Code
* Git
* GitHub

---

## 🔄 Machine Learning Workflow

```text
                 Telco Customer Dataset
                         │
                         ▼
                  Data Understanding
                         │
                         ▼
                   Data Cleaning
                         │
                         ▼
               Exploratory Data Analysis
                         │
                         ▼
                 Feature Engineering
                         │
                         ▼
                 Categorical Encoding
                         │
                         ▼
                  Train/Test Split
                         │
                         ▼
                       SMOTE
                         │
                         ▼
                  Model Training
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Logistic        Decision Tree   Random Forest
    Regression
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                      XGBoost
                         │
                         ▼
                  Model Comparison
                         │
                         ▼
                  Model Evaluation
                         │
                         ▼
              Best Model: Logistic Regression
                         │
                         ▼
                  Model Persistence
                         │
                         ▼
                Streamlit Application
```

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed.

### 1. Missing Value Analysis

The dataset was checked for missing values before model development.

### 2. Data Type Correction

The `TotalCharges` column was converted into a numerical data type.

### 3. Missing Value Handling

Missing values generated during the conversion of `TotalCharges` were handled using the median value.

### 4. Removing Unnecessary Features

The `customerID` column was removed because it is an identifier and does not provide useful predictive information.

### 5. Target Encoding

The `Churn` variable was converted from categorical values into numerical values:

```text
No  → 0
Yes → 1
```

### 6. One-Hot Encoding

Categorical features were converted into numerical features using One-Hot Encoding.

### 7. Train/Test Split

The dataset was divided into training and testing datasets.

The test dataset was kept separate to evaluate the model on unseen data.

### 8. SMOTE

The training dataset contained an imbalance between customers who churned and customers who stayed.

SMOTE was applied **only to the training dataset** to address class imbalance.

This prevents information from the test dataset from leaking into the training process.

---

# 📊 Exploratory Data Analysis

Several analyses were performed to understand customer churn behavior.

## Customer Churn Distribution

![Customer Churn Distribution](results/churn_distribution.png)

The churn distribution shows the proportion of customers who stayed compared with customers who left.

---

## Contract Type vs Churn

![Contract vs Churn](results/contract_vs_churn.png)

Contract type was analyzed to understand its relationship with customer churn.

---

## Key EDA Areas

The project also investigates relationships between churn and:

* Customer tenure
* Monthly charges
* Internet service
* Payment method
* Contract type
* Customer services

These analyses help identify patterns that can be useful for customer retention strategies.

---

# 🤖 Machine Learning Models

Four classification algorithms were considered.

## 1. Logistic Regression

Logistic Regression was used as an interpretable classification model for predicting the probability of customer churn.

It also provides coefficients that can be analyzed to understand the influence of individual features.

---

## 2. Decision Tree

Decision Tree was used to capture non-linear relationships between customer characteristics and churn.

---

## 3. Random Forest

Random Forest combines multiple decision trees to improve prediction performance and reduce the limitations of a single decision tree.

---

## 4. XGBoost

XGBoost was included as a gradient boosting model for comparison with the other classification approaches.

---

# 📈 Model Evaluation

The models were evaluated using multiple metrics rather than relying only on accuracy.

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

For a churn prediction problem, **Recall and F1 Score are particularly important**, because identifying customers who are actually likely to churn is valuable for customer retention efforts.

---

## Model Comparison

![Model Comparison](results/model_comparison.png)

The models were compared using their evaluation results obtained on the unseen test dataset.

---

# 🏆 Final Model

## Logistic Regression

After comparing the implemented classification models, **Logistic Regression was selected as the final model based on the results obtained during experimentation.**

The final model was trained using the SMOTE-balanced training data and evaluated on the original unseen test data.

The trained model is stored in:

```text
models/customer_churn_logistic_regression.pkl
```

---

# 📌 Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

The confusion matrix provides a detailed view of:

* True Positives
* True Negatives
* False Positives
* False Negatives

This helps evaluate how effectively the model distinguishes between customers who churn and customers who stay.

---

# 📈 ROC Curve

![ROC Curve](results/roc_curve.png)

The ROC curve illustrates the model's ability to distinguish between churned and non-churned customers across different classification thresholds.

---

# 🌐 Streamlit Web Application

The project includes an interactive **Streamlit web application**.

The application allows users to enter customer information and receive a churn prediction.

### Application Inputs

The application includes customer information such as:

* Gender
* Senior citizen status
* Partner status
* Dependents
* Tenure
* Phone service
* Multiple lines
* Internet service
* Online security
* Online backup
* Device protection
* Technical support
* Streaming services
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

### Application Output

The application provides:

```text
Churn Probability
       +
Churn Prediction
       +
Risk Level
```

The risk level is categorized as:

```text
LOW RISK
MEDIUM RISK
HIGH RISK
```

---

# ▶️ How to Run the Application

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-prediction.git
```

Move into the project directory:

```bash
cd customer-churn-prediction
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── Data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── customer_churn_logistic_regression.pkl
│   └── model_features.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── results/
│   ├── churn_distribution.png
│   ├── contract_vs_churn.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── src/
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 💾 Saved Model Files

The trained model and feature information are stored in the `models` directory.

### Logistic Regression Model

```text
models/customer_churn_logistic_regression.pkl
```

### Model Feature List

```text
models/model_features.pkl
```

The feature list ensures that prediction data is aligned with the features used during model training.

---

# 📓 Jupyter Notebook

The complete data science experimentation and analysis can be found in:

```text
notebooks/churn_analysis.ipynb
```

The notebook contains:

* Data loading
* Data exploration
* Data cleaning
* EDA
* Feature preprocessing
* SMOTE
* Model training
* Model comparison
* Model evaluation
* Feature analysis

---

# 🔐 Reproducibility

The project includes a `requirements.txt` file containing the required Python libraries.

The scikit-learn version used for the saved model is specified to reduce model compatibility issues.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

# ⚠️ Limitations

This project is intended as an educational and portfolio machine learning project.

Some limitations include:

* The model is trained on historical data.
* Customer behavior can change over time.
* The dataset may not represent every telecom business.
* Model performance may change on different datasets.
* Predictions should be treated as decision-support information rather than guaranteed outcomes.
* The model does not automatically recommend the best retention strategy.

---

# 🚀 Future Improvements

The project can be further improved by implementing:

### Machine Learning

* Hyperparameter tuning
* Cross-validation
* Ensemble model optimization
* Probability threshold optimization
* Calibration of predicted probabilities

### Explainable AI

* SHAP
* LIME
* Feature contribution visualization

### Application

* Customer retention recommendations
* Batch CSV prediction
* Prediction history
* Interactive analytics dashboard
* Customer segmentation
* Model monitoring

### Deployment

* Streamlit Community Cloud
* Docker
* Cloud-based deployment
* REST API integration

---

# 💡 Business Applications

A churn prediction system can help a company:

* Identify high-risk customers
* Prioritize customer retention campaigns
* Offer targeted discounts
* Improve customer support
* Analyze service-related churn patterns
* Reduce customer acquisition costs
* Improve customer lifetime value

For example:

```text
Customer
   ↓
Churn Prediction
   ↓
Risk Assessment
   ↓
High-Risk Customer Identified
   ↓
Retention Strategy
   ↓
Potentially Reduced Churn
```

---

# 🎓 Skills Demonstrated

This project demonstrates practical experience with:

* Python programming
* Data preprocessing
* Exploratory Data Analysis
* Data visualization
* Feature engineering
* Categorical encoding
* Imbalanced classification
* SMOTE
* Logistic Regression
* Decision Trees
* Random Forest
* XGBoost
* Model evaluation
* Classification metrics
* Model persistence
* Streamlit
* Git
* GitHub
* Machine Learning project organization

---

# 👨‍💻 Author

## Siddhesh Dhumal

**B.Tech — Robotics and Automation**

Interested in:

* Data Science
* Machine Learning
* Artificial Intelligence
* Robotics
* Computer Vision

---

# ⭐ Acknowledgement

This project was developed as part of a **Data Science internship project** to gain practical experience in machine learning, data analysis, model evaluation, and application development.

---

## 📌 Project Status

**Status: Completed ✅**

The project includes:

* ✅ Data preprocessing
* ✅ Exploratory Data Analysis
* ✅ SMOTE-based class balancing
* ✅ Multiple ML models
* ✅ Model comparison
* ✅ Logistic Regression final model
* ✅ Model evaluation
* ✅ Saved trained model
* ✅ Streamlit application
* ✅ GitHub-ready project structure

---

⭐ **If you find this project useful, consider giving the repository a star!**
