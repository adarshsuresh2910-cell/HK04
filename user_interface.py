import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Customer Churn Dashboard", layout="wide")

st.title("📊 AI-Assisted Customer Churn Intelligence Dashboard")

# ==============================
# Load Data
# ==============================
df = pd.read_excel("cleaned_telco_data.xlsx")

# ==============================
# 1️⃣ Executive Summary (KPIs)
# ==============================

st.subheader("Executive Summary")

col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
churn_rate = df["Churn Value"].mean() * 100
high_risk_count = len(df[df["Risk Segment"] == "High Risk"])
avg_cltv = df["CLTV"].mean()

col1.metric("Total Customers", total_customers)
col2.metric("Overall Churn Rate (%)", round(churn_rate, 2))
col3.metric("High Risk Customers", high_risk_count)
col4.metric("Average CLTV (₹)", round(avg_cltv, 2))

st.divider()

# ==============================
# 2️⃣ Risk Segmentation
# ==============================

st.subheader("Customer Risk Segmentation")

risk_counts = df["Risk Segment"].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%")
ax1.set_title("Risk Segment Distribution")
st.pyplot(fig1)

# Churn rate by segment
st.write("Churn Rate by Risk Segment:")
st.dataframe(
    df.groupby("Risk Segment")["Churn Value"].mean().reset_index()
)

st.divider()

# ==============================
# 3️⃣ Key Churn Driver
# ==============================

st.subheader("Key Churn Driver: Contract Type")

contract_churn = df.groupby("Contract")["Churn Value"].mean()

fig2, ax2 = plt.subplots()
contract_churn.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Churn Rate")
ax2.set_title("Churn Rate by Contract Type")
st.pyplot(fig2)

st.divider()

# ==============================
# 4️⃣ High-Risk Customer Profile
# ==============================

st.subheader("High-Risk Customer Profile")

high_risk_df = df[df["Risk Segment"] == "High Risk"]

colA, colB, colC, colD = st.columns(4)

colA.metric("Avg Monthly Charges (₹)", round(high_risk_df["Monthly Charges"].mean(), 2))
colB.metric("Avg Tenure (Months)", round(high_risk_df["Tenure Months"].mean(), 2))
colC.metric("Most Common Contract", high_risk_df["Contract"].mode()[0])
colD.metric("Most Common Payment Method", high_risk_df["Payment Method"].mode()[0])

st.divider()

# ==============================
# 5️⃣ High-Risk Customer List
# ==============================

st.subheader("Sample High-Risk Customers")

st.dataframe(
    high_risk_df[[
        "Tenure Months",
        "Monthly Charges",
        "Contract",
        "Payment Method",
        "CLTV",
        "Risk Segment"
    ]].head(20)
)

st.divider()

# ==============================
# 6️⃣ Revenue Impact Estimation
# ==============================

st.subheader("Estimated Revenue Impact")

retention_rate = 0.20
estimated_revenue = len(high_risk_df) * retention_rate * high_risk_df["CLTV"].mean()

st.metric(
    "Estimated Revenue Saved (20% High-Risk Retained)",
    f"₹ {round(estimated_revenue, 2)}"
)

st.success("Targeted retention campaigns focused on High-Risk customers can significantly reduce churn and improve profitability.")