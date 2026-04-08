import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Churn Dashboard", page_icon="📊", layout="wide")

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

pipe = pickle.load(open("churn_system.pkl", "rb"))

df = pd.read_csv("Telecom_churn.csv")

df['Churn_binary'] = df['Churn'].map({'No':0,'Yes':1})
df['Contract'] = df['Contract'].fillna("Month-to-month")
df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].median())
df['tenure'] = df['tenure'].fillna(df['tenure'].median())

page = st.sidebar.selectbox("Navigation", ["Dashboard", "Predict Churn"])

if page == "Dashboard":
    st.title("Customer Churn Dashboard")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "≈ 82%")
    col2.metric("Model Type", "Stacking Classifier")
    col3.metric("Goal", "Reduce Churn")
    
    st.markdown("---")
    st.subheader("Example Insights")
    st.write("- Most churn occurs for Month-to-Month contracts")
    st.write("- Customers with low tenure are more likely to churn")
    st.write("- High MonthlyCharges increase churn risk")
    
    st.markdown("### Churn by Contract Type")
    churn_contract = df.groupby("Contract")["Churn_binary"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=churn_contract.index, y=churn_contract.values, palette="coolwarm", ax=ax)
    ax.set_ylabel("Churn Rate")
    st.pyplot(fig)
    
    st.markdown("### Churn by Tenure")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", palette="Set2", bins=20, ax=ax2)
    st.pyplot(fig2)
    
    st.markdown("### Monthly Charges Distribution by Churn")
    fig3, ax3 = plt.subplots()
    sns.histplot(data=df, x="MonthlyCharges", hue="Churn", multiple="stack", palette="Set1", bins=20, ax=ax3)
    st.pyplot(fig3)
    
    st.markdown("### InternetService impact on churn")
    fig4, ax4 = plt.subplots()
    sns.histplot(data=df, x="InternetService", hue="Churn", multiple="stack", palette='Set2', bins=20, ax=ax4)
    st.pyplot(fig4)
    st.markdown("### SeniorCitizen impact on churn")
    fig5, ax5 = plt.subplots()
    sns.histplot(data=df, x="SeniorCitizen", hue="Churn", multiple="stack", palette='Set2', bins=20, ax=ax5)
    st.pyplot(fig5)

    st.markdown("### family on churn")
    fig6, ax6 = plt.subplots()
    sns.histplot(data=df, x="OnlineSecurity", hue="Churn", multiple="stack", palette='Set2', bins=20, ax=ax6)
    st.pyplot(fig6)

    st.markdown("### MultipleLines impact on churn")
    fig7, ax7 = plt.subplots()
    sns.histplot(data=df, x="MultipleLines", hue="Churn", multiple="stack", palette='Set2', bins=20, ax=ax7)
    st.pyplot(fig7)

    st.markdown("### OnlineBackup impact on churn")
    fig8, ax8 = plt.subplots()
    sns.histplot(data=df, x="OnlineBackup", hue="Churn", multiple="stack", palette='Set2', bins=20, ax=ax8)
    st.pyplot(fig8)

    st.markdown("### DeviceProtection impact on churn")
    fig8, ax8 = plt.subplots()
    sns.histplot(data=df, x="DeviceProtection", hue="Churn", multiple="stack", palette='Set2', bins=20, ax=ax8)
    st.pyplot(fig8)
    
   
if page == "Predict Churn":
    st.title("Predict Customer Churn")
    
    st.subheader("Customer Details")
    col1, col2 = st.columns(2)
    with col1:
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with col2:
        MonthlyCharges = st.slider("Monthly Charges", 0, 200, 50)
        TotalCharges = st.number_input("Total Charges", 0.0)
    
    st.subheader("📡 Services")
    col3, col4 = st.columns(2)
    with col3:
        InternetService = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service","No","Yes"])
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    with col4:
        OnlineSecurity = st.selectbox("Online Security", ["No","Yes","No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No","Yes","No internet service"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"
        ])
    
    st.subheader("Extra Features")
    col5, col6 = st.columns(2)
    with col5:
        StreamingTV = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
    with col6:
        DeviceProtection = st.selectbox("Device Protection", ["No","Yes","No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
    
    st.markdown("---")
    
    if st.button("Predict"):
        Partner_bin = 1 if Partner=="Yes" else 0
        Dependents_bin = 1 if Dependents=="Yes" else 0
        family = Partner_bin + Dependents_bin + 1
        MultipleLines = {"No phone service":0, "No":1, "Yes":2}[MultipleLines]
        def bin_map(x): return 1 if x=="Yes" else 0
        OnlineSecurity = 0 if OnlineSecurity=="No internet service" else bin_map(OnlineSecurity)
        OnlineBackup = 0 if OnlineBackup=="No internet service" else bin_map(OnlineBackup)
        DeviceProtection = 0 if DeviceProtection=="No internet service" else bin_map(DeviceProtection)
        TechSupport = 0 if TechSupport=="No internet service" else bin_map(TechSupport)
        StreamingTV_bin = 0 if StreamingTV=="No internet service" else bin_map(StreamingTV)
        StreamingMovies_bin = 0 if StreamingMovies=="No internet service" else bin_map(StreamingMovies)
        Entertainment = StreamingTV_bin | StreamingMovies_bin
        PaperlessBilling = 1 if PaperlessBilling=="Yes" else 0
        
        input_df = pd.DataFrame([{
            "SeniorCitizen": SeniorCitizen,
            "family": family,
            "tenure": tenure,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "Entertainment": Entertainment,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])
        
        pred = pipe.predict(input_df)
        prob = pipe.predict_proba(input_df)[0][1]
        
        st.subheader("Prediction Result")
        if pred[0]==1:
            st.error(f"High Churn Risk ({round(prob*100,2)}%)")
        else:
            st.success(f"Low Churn Risk ({round((1-prob)*100,2)}%)")
            st.balloons()
        
        st.subheader("💡 Recommendations")
        suggestions = []
        if tenure<6: suggestions.append("Offer onboarding discounts")
        if MonthlyCharges>80: suggestions.append("Suggest cheaper plan")
        if Contract=="Month-to-month": suggestions.append("Encourage long-term contract")
        if TechSupport==0: suggestions.append("Provide tech support")
        if Entertainment==0: suggestions.append("Offer streaming services")
        if suggestions:
            for s in suggestions: st.write("->", s)
        else:
            st.write("Customer is stable")
