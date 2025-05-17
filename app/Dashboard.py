import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Dashboard")

st.title("StockSentinel")
st.subheader("_A brief:_")
st.write("StockSentinel is an attempt to push the boundaries of the automated stock trading models. "
"As stock trading models utilize Deep Reinforcement Learning (DRL) models, like A2C, DDPG, PPO, etc., we implemented Stock Trading model for both prediction and trading. "
"For predicting the future price based on the metrics LSTM (Long Short-Term Memory), a DL model, was used, and for autonomus trading PPO (Proximal Policy Optimization), a DRL algorithm, was used. "
"And stock metrics like EMA_12, EMA26, MACD, Signal, RSI, CCI, ADX were used for better prediction and trading. "
"Also, financial and business news were utilized, where a sentiment score was given by FinBERT model. ")

st.subheader("_Architecture:_")
st.image("./app/pages/MajorProject_background.png", caption="Architecture of the project")
st.image("./app/pages/MajorProject_hybrid_background.png", caption="Architecture of the Hybrid Model")

st.subheader("_Results_")
st.info("Note: This is only comparision with DJIA (Dow Jones Industrial Average).")
data = {
    "Metrics": [
        "Cumulative Returns", "Max Earning Rate", "Max Pullbacks",
        "Average Profitability Per Trade", "Sharpe Ratio"
    ],
    "Hybrid (Ours)": [97.438588, 111.343139, 11.795118, 89.147839, 2.207112],
    "PPO": [54.3700, 67.2800, 28.3000, 20.0200, 0.8081],
    "RecurrentPPO": [49.7700, 63.4500, 29.8900, 22.8400, 0.6819],
    "CLSTM-PPO": [90.810, 113.500, 46.510, 35.270, 1.154],
    "Ensemble": [70.40, 65.32, 15.74, 28.84, 1.30],
    "DJI": [50.9700, 63.3000, 72.3200, 0.0000, 0.4149],
}

# Create DataFrame
df = pd.DataFrame(data)

# Display title and table
st.subheader("Performance Metrics Table")
st.dataframe(df)

# Transpose for easier plotting
df_melted = df.melt(id_vars=["Metrics"], var_name="Model", value_name="Value")

# Bar chart using Plotly
st.subheader("Metric Comparison Bar Charts")
for metric in df["Metrics"]:
    fig = px.bar(
        df_melted[df_melted["Metrics"] == metric],
        x="Model",
        y="Value",
        title=metric,
        color="Model",
        text="Value"
    )
    fig.update_layout(xaxis_title="Model", yaxis_title=metric)
    st.plotly_chart(fig, use_container_width=True)