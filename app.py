import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import time
import joblib

# ====================== CONFIG ======================
st.set_page_config(
    page_title="Solar Intelligence System", 
    layout="wide",
    page_icon="☀️"
)

# ====================== DATABASE CONNECTION ======================
MONGO_URI = "mongodb+srv://ThanhTam:Tamdo1911@clusterbigdata.up9klu6.mongodb.net/"

@st.cache_resource
def init_connection():
    return pymongo.MongoClient(MONGO_URI)

client = init_connection()
db = client['SolarProject']

# ====================== LOAD SCALERS ======================
@st.cache_resource
def load_scalers():
    scaler_minmax = joblib.load('scaler_minmax.joblib')
    scaler_standard = joblib.load('scaler_standard.joblib')
    return scaler_minmax, scaler_standard

scaler_minmax, scaler_standard = load_scalers()

# ====================== LOAD DATA ======================
@st.cache_data(ttl=60)
def load_enhanced_data():
    perf_raw = list(db['inverter_performance_daily'].find())
    weather_raw = list(db['weather_power_forecast'].find())
    alerts_raw = list(db['system_alerts'].find())
    
    df_perf = pd.json_normalize(perf_raw) if perf_raw else pd.DataFrame()
    df_weather = pd.json_normalize(weather_raw) if weather_raw else pd.DataFrame()
    df_alerts = pd.json_normalize(alerts_raw) if alerts_raw else pd.DataFrame()
    
    return df_perf, df_weather, df_alerts

df_perf, df_weather, df_alerts = load_enhanced_data()

# ====================== SIDEBAR ======================
st.sidebar.title("🔧 Control Center")
selected_plant = st.sidebar.selectbox(
    "Select Plant Unit", 
    options=["All Units"] + list(df_perf['plant_id'].unique()) if not df_perf.empty else ["All Units"]
)

# Filter data
filtered_perf = df_perf.copy()
filtered_weather = df_weather.copy()
if selected_plant != "All Units":
    filtered_perf = filtered_perf[filtered_perf['plant_id'] == selected_plant]
    filtered_weather = filtered_weather[filtered_weather['plant_id'] == selected_plant]

# ====================== MAIN TITLE ======================
st.header("☀️ Solar Power Generation & Optimization Intelligence")

tabs = st.tabs(["Performance Metrics", "Operational Strategy", "Asset Health", "Deep Analytics", "⚡ Real-time Monitoring"])

# ====================== TAB 1: PERFORMANCE METRICS (ĐÃ CÓ BIỂU ĐỒ KẾT HỢP) ======================
with tabs[0]:
    st.subheader("📊 KPI & Yield Analysis")
    
    if not filtered_perf.empty:
        total_act = filtered_perf['metrics.actual_daily_yield'].sum()
        total_pre = filtered_perf['metrics.predicted_daily_yield'].sum()
        gap = total_act - total_pre

        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Actual Yield", f"{total_act:,.1f} kWh")
        col2.metric("Total AI Prediction", f"{total_pre:,.1f} kWh")
        col3.metric("Variance (Gap)", f"{gap:,.1f} kWh", delta=f"{gap:,.1f} kWh")
        col4.metric("Avg Efficiency", f"{filtered_perf['metrics.avg_conversion_efficiency'].mean():.2%}")

        st.divider()

        # Waterfall Chart
        st.write("**Waterfall: AI Prediction → Actual Yield**")
        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["AI Prediction", "Yield Gap", "Actual Result"],
            text=[f"{total_pre:,.0f}", f"{gap:,.0f}", f"{total_act:,.0f}"],
            y=[total_pre, gap, total_act],
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            totals={"marker": {"color": "#3498db"}}
        ))
        fig_wf.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig_wf, use_container_width=True)

        st.divider()

        # ==================== BIỂU ĐỒ KẾT HỢP THEO INVERTER ====================
        st.subheader("⚖️ So sánh Sản lượng theo từng Inverter")

        df_inv = filtered_perf.groupby('source_key').agg({
            'metrics.actual_daily_yield': 'sum',
            'metrics.predicted_daily_yield': 'sum'
        }).reset_index()

        df_inv = df_inv.sort_values('metrics.actual_daily_yield', ascending=False)

        fig_combined = go.Figure()

        # Cột Actual Yield
        fig_combined.add_trace(go.Bar(
            x=df_inv['source_key'],
            y=df_inv['metrics.actual_daily_yield'],
            name='Actual Yield (Thực tế)',
            marker_color='#3498db',
            text=df_inv['metrics.actual_daily_yield'].round(1),
            textposition='auto'
        ))

        # Đường AI Predicted
        fig_combined.add_trace(go.Scatter(
            x=df_inv['source_key'],
            y=df_inv['metrics.predicted_daily_yield'],
            name='AI Predicted Target',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=4),
            marker=dict(size=10)
        ))

        fig_combined.update_layout(
            title="So sánh Sản lượng Thực tế vs AI Dự báo theo từng Inverter",
            template="plotly_white",
            height=550,
            xaxis_title="Inverter ID",
            yaxis_title="Daily Yield (kWh)",
            legend=dict(orientation="h", y=1.12),
            barmode='group'
        )

        st.plotly_chart(fig_combined, use_container_width=True)

    else:
        st.warning("Không có dữ liệu hiệu suất để hiển thị.")

# ====================== TAB 2: Operational Strategy ======================
with tabs[1]:
    st.subheader("Environmental Correlation & Strategy")
    if not filtered_weather.empty:
        fig_opt = px.scatter(
            filtered_weather, 
            x="features_snapshot.irradiation", 
            y="predictions.expected_ac_power", 
            color="weather_condition", 
            title="Irradiation vs AC Generation Analysis"
        )
        fig_opt.update_layout(template="plotly_white")
        st.plotly_chart(fig_opt, use_container_width=True)

    st.divider()
    st.subheader("AI Generation Simulator (What-If Analysis)")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        sim_irr = st.slider("Irradiation Level (W/m²)", 0.0, 1.2, 0.8, 0.05)
        sim_temp = st.slider("Module Temperature (°C)", 20.0, 65.0, 35.0, 1.0)
        sim_weather = st.selectbox("Weather Scenario", ["Sunny", "Partly Cloudy", "Cloudy"])
        
        base_eff, area, temp_coeff = 0.18, 8000, 0.004
        expected_yield = sim_irr * area * base_eff * (1 - temp_coeff * (sim_temp - 25))
        if sim_weather == "Cloudy": expected_yield *= 0.7
        elif sim_weather == "Partly Cloudy": expected_yield *= 0.9

    with sc2:
        fig_sim = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=expected_yield, 
            title={'text': "Predicted Power (kW)"},
            gauge={'axis': {'range': [None, 2000]}, 
                   'bar': {'color': "#3498db"}}
        ))
        fig_sim.update_layout(height=350)
        st.plotly_chart(fig_sim, use_container_width=True)

# ====================== TAB 3: Asset Health ======================
with tabs[2]:
    st.subheader("Asset Integrity & Maintenance Log")
    if not filtered_perf.empty:
        fig_health = px.bar(
            filtered_perf, 
            x="source_key", 
            y="classification.health_score", 
            color="classification.status", 
            title="Inverter Health Score",
            range_y=[0, 100],
            color_discrete_map={'Good': '#27ae60', 'Needs Attention': '#e67e22'}
        )
        fig_health.update_layout(template="plotly_white")
        st.plotly_chart(fig_health, use_container_width=True)
    
    if not df_alerts
