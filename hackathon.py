
import pandas as pd
import numpy as np

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_excel("Telco_customer_churn.xlsx", engine="openpyxl")
print("Dataset Loaded Successfully")

# Clean column names (remove spaces issues)
df.columns = df.columns.str.strip()

# ==============================
# 2. DROP IRRELEVANT COLUMNS
# ==============================
columns_to_drop = [
    "CustomerID", "Count", "Country", "State",
    "City", "Zip Code", "Lat Long",
    "Latitude", "Longitude"
]

df.drop(columns=columns_to_drop, errors="ignore", inplace=True)

# ==============================
# 3. DATA CLEANING
# ==============================

# Convert Total Charges to numeric
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"].fillna(df["Total Charges"].median(), inplace=True)

# Convert Yes/No to 1/0
binary_map = {"Yes": 1, "No": 0}

binary_columns = [
    "Partner", "Dependents", "Phone Service",
    "Paperless Billing"
]

for col in binary_columns:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

# Convert Senior Citizen (0/1 already numeric usually)
# Clean Senior Citizen safely
df["Senior Citizen"] = df["Senior Citizen"].replace({"Yes": 1, "No": 0})
df["Senior Citizen"] = pd.to_numeric(df["Senior Citizen"], errors="coerce")
df["Senior Citizen"].fillna(0, inplace=True)

# ==============================
# 4. FEATURE ENGINEERING
# ==============================

# Tenure Groups
df["Tenure Group"] = pd.cut(
    df["Tenure Months"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
)

# High Monthly Charge Flag
df["High Monthly Charge"] = np.where(
    df["Monthly Charges"] > df["Monthly Charges"].median(),
    1,
    0
)

# Average Monthly Spend
df["Avg Monthly Spend"] = df["Total Charges"] / (df["Tenure Months"] + 1)

# Contract Risk (Month-to-month risky)
df["Contract Risk"] = df["Contract"].apply(
    lambda x: 2 if x == "Month-to-month" else 1
)

# Complaint Flag (Based on Churn Reason)
#df["Has Complaint"] = np.where(df["Churn Reason"].notna(), 1, 0)

# ==============================
# 5. CHURN DRIVER ANALYSIS
# ==============================

print("\nChurn Rate by Contract:")
print(df.groupby("Contract")["Churn Value"].mean())

print("\nChurn Rate by Tenure Group:")
print(df.groupby("Tenure Group")["Churn Value"].mean())

print("\nChurn Rate by Payment Method:")
print(df.groupby("Payment Method")["Churn Value"].mean())

print("\nTop Correlations with Churn:")
print(df.corr(numeric_only=True)["Churn Value"].sort_values(ascending=False))

# ==============================
# 6. RISK SEGMENTATION
# ==============================

def risk_segment(score):
    if score >= 75:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk Segment"] = df["Churn Score"].apply(risk_segment)

print("\nRisk Segment Distribution:")
print(df["Risk Segment"].value_counts())

# ==============================
# 7. SAVE CLEANED FILE
# ==============================
df.to_excel("cleaned_telco_data.xlsx", index=False)
print("\nCleaned dataset saved successfully.")

print("\nChurn Rate by Risk Segment:")

print(df.groupby("Risk Segment")["Churn Value"].mean())
