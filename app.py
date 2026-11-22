import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Vantara Customer Intelligence", page_icon="📊", layout="wide")

st.title("📊 Vantara Customer Intelligence Platform")
st.markdown("AI-Powered Customer Segmentation, Loyalty Analysis & Live Prediction Dashboard")

@st.cache_data
def load_data():
    np.random.seed(42)
    num_customers = 250
    data = {
        'Customer_ID': [f"CUST-{1000+i}" for i in range(num_customers)],
        'Age': np.random.randint(18, 65, num_customers),
        'Annual_Income_k': np.random.randint(15, 140, num_customers),
        'Spending_Score': np.random.randint(1, 100, num_customers),
        'Purchase_Frequency': np.random.randint(1, 20, num_customers)
    }
    df = pd.DataFrame(data)
    
    def assign_segment(row):
        inc = row['Annual_Income_k']
        score = row['Spending_Score']
        if inc >= 70 and score >= 60:
            return 'VIP / High-Value Targets'
        elif inc >= 70 and score < 60:
            return 'Careful High-Earners'
        elif inc < 70 and score >= 60:
            return 'Impulsive Spenders'
        elif inc >= 40:
            return 'Standard Customers'
        else:
            return 'Budget-Conscious Buyers'

    df['Segment'] = df.apply(assign_segment, axis=1)
    df['Loyalty_Index'] = (df['Spending_Score'] * 0.6 + df['Purchase_Frequency'] * 2).round(1)
    return df

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Avg Annual Income", f"${df['Annual_Income_k'].mean():.1f}k")
col3.metric("Avg Spending Score", f"{df['Spending_Score'].mean():.1f} / 100")
col4.metric("VIP Customers", len(df[df['Segment'] == 'VIP / High-Value Targets']))

st.divider()

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🎯 AI Customer Segmentation Map")
    st.scatter_chart(df, x='Annual_Income_k', y='Spending_Score', color='Segment')

with right_col:
    st.subheader("📈 Customer Count by Segment")
    st.bar_chart(df['Segment'].value_counts())

st.divider()

st.sidebar.header("🤖 Live AI Customer Predictor")
st.sidebar.write("Enter customer details to predict segment:")

in_income = st.sidebar.slider("Annual Income ($k)", 15, 140, 75)
in_score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 80)

if st.sidebar.button("Predict Customer Segment"):
    if in_income >= 70 and in_score >= 60:
        pred_label = 'VIP / High-Value Targets'
        strategy = "Offer exclusive loyalty rewards and VIP early access."
    elif in_income >= 70 and in_score < 60:
        pred_label = 'Careful High-Earners'
        strategy = "Highlight premium product quality and value bundles."
    elif in_income < 70 and in_score >= 60:
        pred_label = 'Impulsive Spenders'
        strategy = "Target with flash sales and trend-based promotions."
    elif in_income >= 40:
        pred_label = 'Standard Customers'
        strategy = "Engage with regular newsletters and combo offers."
    else:
        pred_label = 'Budget-Conscious Buyers'
        strategy = "Target with seasonal discounts and promotional coupons."
        
    st.sidebar.success(f"**Category:** {pred_label}")
    st.sidebar.info(f"💡 **Strategy:** {strategy}")

st.subheader("📋 Customer Intelligence Data Report")
st.dataframe(df.head(15), use_container_width=True)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Full AI Report (CSV)", data=csv, file_name="vantara_customer_report.csv", mime="text/csv")

    
   


       
