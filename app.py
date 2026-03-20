import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# ==========================================
# 1. PAGE SETUP & ASSET LOADING
# ==========================================
st.set_page_config(page_title="Bank Churn Predictor", layout="wide")
st.title("🏦 Customer Churn Risk Dashboard")
st.markdown("Real-time churn predictions, risk factor analysis, and what-if simulations.")

@st.cache_resource 
def load_assets():
    model = joblib.load('best_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    # Automatically load your dataset here
    # Replace 'your_data.csv' with your actual filename
    raw_data = pd.read_csv('European_Bank.csv') 
    return model, scaler, feature_cols, raw_data

model, scaler, feature_cols, raw_df = load_assets()

# ==========================================
# 2. PREPROCESSING LOGIC
# ==========================================
def preprocess_data(df, feature_cols):
    df_clean = df.copy()
    cols_to_drop = [c for c in ['CustomerId', 'Surname', 'Exited'] if c in df_clean.columns]
    df_clean = df_clean.drop(cols_to_drop, axis=1)
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    df_clean.fillna('Unknown', inplace=True)
    df_clean = pd.get_dummies(df_clean, columns=['Geography', 'Gender'], drop_first=True)
    
    if 'Balance' in df_clean.columns and 'EstimatedSalary' in df_clean.columns:
        df_clean['Balance_to_Salary'] = df_clean['Balance'] / df_clean['EstimatedSalary']
        df_clean['Balance_to_Salary'].replace([np.inf, -np.inf], 0, inplace=True) 
        
    if 'NumOfProducts' in df_clean.columns and 'Tenure' in df_clean.columns:
        df_clean['Product_Density'] = df_clean['NumOfProducts'] / (df_clean['Tenure'] + 0.01)
        
    if 'IsActiveMember' in df_clean.columns and 'NumOfProducts' in df_clean.columns:
        df_clean['Engagement_Product'] = df_clean['IsActiveMember'] * df_clean['NumOfProducts']
        
    if 'Age' in df_clean.columns and 'Tenure' in df_clean.columns:
        df_clean['Age_Tenure'] = df_clean['Age'] * df_clean['Tenure']

    df_clean = df_clean.reindex(columns=feature_cols, fill_value=0)
    return df_clean

# ==========================================
# 3. AUTOMATIC PREDICTIONS & INSIGHTS
# ==========================================
st.header("📊 Global Churn Insights")

# Run predictions on the loaded dataset immediately
processed_df = preprocess_data(raw_df, feature_cols)
scaled_data = scaler.transform(processed_df)
predictions = model.predict(scaled_data)
probabilities = model.predict_proba(scaled_data)[:, 1]

results_df = raw_df.copy()
results_df['Churn_Probability'] = probabilities
results_df['Predicted_Churn'] = predictions

# UI Layout for Insights
col_table, col_viz = st.columns([1.2, 0.8])

with col_table:
    st.subheader("Customer Prediction List")
    st.dataframe(results_df[['CustomerId', 'Surname', 'Churn_Probability', 'Predicted_Churn']], height=400)

with col_viz:
    st.subheader("Risk Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probabilities, bins=20, color='coral', edgecolor='white')
    ax.set_title('Churn Probabilities across Dataset')
    st.pyplot(fig)

# SHAP Insights (Global)
st.subheader("🧠 Why are customers leaving? (Feature Importance)")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(scaled_data)
if isinstance(shap_values, list): shap_values = shap_values[1]

fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values, processed_df, show=False)
st.pyplot(fig_shap)

st.divider()

# ==========================================
# 4. WHAT-IF SCENARIO SIMULATOR
# ==========================================
st.header("🎛️ Individual What-If Simulator")
col_input, col_result = st.columns([1, 1])

with col_input:
    st.subheader("Input customer features")
    year = st.number_input("Year", min_value=2000, value=2025) 
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0)
    salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0)
    num_products = st.slider("Number of Products", 1, 4, 2)
    is_active = st.selectbox("Is Active Member?", [0, 1], index=1)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

with col_result:
    st.subheader("Simulation Results")
    
    # Feature Engineering for simulation
    sim_data_dict = {
        'Year': year, 'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
        'Balance': balance, 'NumOfProducts': num_products, 'HasCrCard': 1,
        'IsActiveMember': is_active, 'EstimatedSalary': salary,
        'Geography_Germany': 1 if geography == "Germany" else 0,
        'Geography_Spain': 1 if geography == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Balance_to_Salary': balance / salary if salary > 0 else 0,
        'Product_Density': num_products / (tenure + 0.01),
        'Engagement_Product': is_active * num_products,
        'Age_Tenure': age * tenure
    }
    
    sim_data = pd.DataFrame([sim_data_dict])[feature_cols]
    sim_prob = model.predict_proba(scaler.transform(sim_data))[0][1]
    
    st.metric(label="Simulated Churn Probability", value=f"{sim_prob * 100:.2f}%")
    if sim_prob > 0.5:
        st.error("🚨 HIGH RISK: This configuration likely leads to CHURN.")
    else:
        st.success("✅ LOW RISK: This configuration suggests RETENTION.")