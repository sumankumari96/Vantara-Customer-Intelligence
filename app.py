import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    return pd.DataFrame(data)

df = load_data()

features = df[['Annual_Income_k', 'Spending_Score']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

segment_map = {
    0: 'Standard Customers',
    1: 'VIP / High-Value Targets',
    2: 'Careful High-Earners',
    3: 'Impulsive Spenders',
    4: 'Budget-Conscious Buyers'
}
df['Segment'] = df['Cluster'].map(segment_map)
df['Loyalty_Index'] = (df['Spending_Score'] * 0.6 + df['Purchase_Frequency'] * 2).round(1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(df))
col2.metric("Avg Annual Income", f"${df['Annual_Income_k'].mean():.1f}k")
col3.metric("Avg Spending Score", f"{df['Spending_Score'].mean():.1f} / 100")
col4.metric("VIP Customers", len(df[df['Segment'] == 'VIP / High-Value Targets']))

st.divider()

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🎯 AI Customer Segmentation Map")
    fig = px.scatter(
        df, x='Annual_Income_k', y='Spending_Score', color='Segment',
        hover_data=['Customer_ID', 'Age', 'Loyalty_Index'],
        title="Income vs. Spending Behavior"
    )
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("📈 Segment Distribution")
    pie_fig = px.pie(df, names='Segment', hole=0.4)
    st.plotly_chart(pie_fig, use_container_width=True)

st.divider()

st.sidebar.header("🤖 Live AI Customer Predictor")
st.sidebar.write("Enter customer details to predict segment:")

in_income = st.sidebar.slider("Annual Income ($k)", 15, 140, 75)
in_score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 80)

if st.sidebar.button("Predict Customer Segment"):
    user_scaled = scaler.transform([[in_income, in_score]])
    pred_cluster = kmeans.predict(user_scaled)[0]
    pred_label = segment_map[pred_cluster]
    
    st.sidebar.success(f"**Category:** {pred_label}")
    if pred_label == 'VIP / High-Value Targets':
        st.sidebar.info("💡 **Strategy:** Offer exclusive loyalty rewards and VIP early access.")
    elif pred_label == 'Careful High-Earners':
        st.sidebar.info("💡 **Strategy:** Highlight premium product quality and value bundles.")
    else:
        st.sidebar.info("💡 **Strategy:** Target with seasonal discounts and promotional coupons.")

st.subheader("📋 Customer Intelligence Data Report")
st.dataframe(df.head(15), use_container_width=True)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Full AI Report (CSV)", data=csv, file_name="vantara_customer_report.csv", mime="text/csv")
