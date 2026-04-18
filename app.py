%%writefile solar_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import time
import joblib

# 1. Database Connection
MONGO_URI = "mongodb+srv://ThanhTam:Tamdo1911@clusterbigdata.up9klu6.mongodb.net/"
@st.cache_resource
def init_connection():
    return pymongo.MongoClient(MONGO_URI)

client = init_connection()
db = client['SolarProject']

@st.cache_resource
def load_scalers():
    scaler_minmax = joblib.load('scaler_minmax.joblib')
    scaler_standard = joblib.load('scaler_standard.joblib')
    return scaler_minmax, scaler_standard

scaler_minmax, scaler_standard = load_scalers()

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

# 3. UI Configuration
st.set_page_config(page_title="Solar Intelligence System", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 5px; border: 1px solid #e1e4e8; }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("Control Center")
selected_plant = st.sidebar.selectbox("Select Plant Unit", options=["All Units"] + list(df_perf['plant_id'].unique()) if not df_perf.empty else ["All Units"])

filtered_perf = df_perf.copy()
filtered_weather = df_weather.copy()
if selected_plant != "All Units":
    filtered_perf = filtered_perf[filtered_perf['plant_id'] == selected_plant]
    filtered_weather = filtered_weather[filtered_weather['plant_id'] == selected_plant]

st.header("Solar Power Generation & Optimization Intelligence")
tabs = st.tabs(["Performance Metrics", "Operational Strategy", "Asset Health", "Deep Analytics", "⚡ Real-time Monitoring"])

with tabs[0]:
    st.subheader("📊 KPI & Yield Analysis")
    if not filtered_perf.empty:
        total_act = filtered_perf['metrics.actual_daily_yield'].sum()
        total_pre = filtered_perf['metrics.predicted_daily_yield'].sum()
        gap = total_act - total_pre

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Actual Yield", f"{total_act:,.1f} kWh")
        m2.metric("Predictive Baseline", f"{total_pre:,.1f} kWh")
        m3.metric("Variance (Gap)", f"{gap:,.1f} kWh", delta=f"{gap:,.1f}")
        m4.metric("Avg Efficiency", f"{filtered_perf['metrics.avg_conversion_efficiency'].mean():.2%}")

        st.write("**Waterfall: Predictive to Actual Flow**")
        fig_wf = go.Figure(go.Waterfall(
            name = "Yield Gap", orientation = "v",
            measure = ["absolute", "relative", "total"],
            x = ["AI Prediction", "Yield Gap", "Actual Result"],
            textposition = "outside",
            text = [f"{total_pre:,.0f}", f"{gap:,.0f}", f"{total_act:,.0f}"],
            y = [total_pre, gap, total_act],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            increasing = {"marker":{"color":"#2ecc71"}},
            decreasing = {"marker":{"color":"#e74c3c"}},
            totals = {"marker":{"color":"#3498db"}}
        ))
        fig_wf.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_wf, use_container_width=True)

        st.divider()
        st.write("**Inverter Benchmark: Actual vs AI Target**")
        df_inv = filtered_perf.groupby('source_key').agg({'metrics.actual_daily_yield':'sum', 'metrics.predicted_daily_yield':'sum'}).reset_index()
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Bar(x=df_inv['source_key'], y=df_inv['metrics.actual_daily_yield'], name='Actual', marker_color='#3498db'))
        fig_ai.add_trace(go.Scatter(x=df_inv['source_key'], y=df_inv['metrics.predicted_daily_yield'], mode='lines+markers', name='AI Baseline', line=dict(color='#e74c3c', width=2)))
        fig_ai.update_layout(template="plotly_white", height=500, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_ai, use_container_width=True)

with tabs[1]:
    st.subheader("Environmental Correlation & Strategy")
    if not filtered_weather.empty:
        fig_opt = px.scatter(filtered_weather, x="features_snapshot.irradiation", y="predictions.expected_ac_power", color="weather_condition", trendline="ols", title="Irradiation vs AC Generation Analysis")
        fig_opt.update_layout(template="plotly_white")
        st.plotly_chart(fig_opt, use_container_width=True)

    st.divider()
    st.subheader("AI Generation Simulator (What-If Analysis)")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.write("**Adjust Environmental Parameters**")
        sim_irr = st.slider("Irradiation Level (W/m²)", 0.0, 1.2, 0.8, 0.05)
        sim_temp = st.slider("Module Temperature (°C)", 20.0, 65.0, 35.0, 1.0)
        sim_weather = st.selectbox("Weather Scenario", ["Sunny", "Partly Cloudy", "Cloudy"])

        base_eff, area, temp_coeff = 0.18, 8000, 0.004
        expected_yield = sim_irr * area * base_eff * (1 - temp_coeff * (sim_temp - 25))
        if sim_weather == "Cloudy": expected_yield *= 0.7
        elif sim_weather == "Partly Cloudy": expected_yield *= 0.9

    with sc2:
        fig_sim = go.Figure(go.Indicator(
            mode = "gauge+number", value = expected_yield, title = {'text': "Predicted Power (kW)"},
            gauge = {'axis': {'range': [None, 2000]}, 'bar': {'color': "#3498db"}, 'threshold' : {'line': {'color': "red", 'width': 4}, 'value': 1800}}
        ))
        fig_sim.update_layout(height=350)
        st.plotly_chart(fig_sim, use_container_width=True)

with tabs[2]:
    st.subheader("Asset Integrity & Maintenance Log")
    if not filtered_perf.empty:
        fig_health = px.bar(filtered_perf, x="source_key", y="classification.health_score", color="classification.status", title="Inverter Health Score", range_y=[0, 100], color_discrete_map={'Good': '#27ae60', 'Needs Attention': '#e67e22'})
        fig_health.update_layout(template="plotly_white")
        st.plotly_chart(fig_health, use_container_width=True)
    if not df_alerts.empty:
        st.table(df_alerts[['source_key', 'severity', 'root_cause_analysis.suggested_action']].head(10))

with tabs[3]:
    st.subheader("Advanced Predictive Analytics")
    if not filtered_weather.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_ts = px.area(filtered_weather.sort_values('timestamp'), x="timestamp", y="predictions.expected_ac_power", title="Projected Generation Potential", color_discrete_sequence=['#3498db'])
            st.plotly_chart(fig_ts, use_container_width=True)
        with col2:
            fig_hm = px.density_heatmap(filtered_weather, x="features_snapshot.module_temp", y="features_snapshot.irradiation", z="predictions.expected_ac_power", histfunc="avg", title="Density Heatmap", color_continuous_scale="Viridis")
            st.plotly_chart(fig_hm, use_container_width=True)

with tabs[4]:
    st.subheader("⚡ Live Data Stream Simulator")
    # FIXED: Added allow_disk_use=True to bypass MongoDB sort memory limit
    rt_cursor = db['realtime_feeds'].find().sort("original_ts", 1).allow_disk_use(True)
    df_rt_raw = pd.DataFrame(list(rt_cursor))
    if not df_rt_raw.empty:
        scaling_cols = ['DC_POWER', 'AC_POWER', 'IRRADIATION', 'DC_POWER_lag_1', 'DC_POWER_lag_4', 'IRRADIATION_lag_1', 'IRRADIATION_lag_4', 'DC_POWER_rolling_4', 'IRRADIATION_rolling_4']
        std_cols = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'Temp_Diff', 'avg_conversion_efficiency']

        for col in ['dc_power', 'ac_power', 'irradiation']:
            idx = scaling_cols.index(col.upper())
            m_min = scaler_minmax.min_[idx]
            m_scale = scaler_minmax.scale_[idx]
            df_rt_raw[col] = np.expm1((df_rt_raw[col] - m_min) / m_scale)

        t_idx = std_cols.index('MODULE_TEMPERATURE')
        df_rt_raw['module_temp'] = (df_rt_raw['module_temp'] * scaler_standard.scale_[t_idx]) + scaler_standard.mean_[t_idx]

        df_rt = df_rt_raw.groupby('original_ts').agg({
            'ac_power': 'mean',
            'dc_power': 'mean',
            'irradiation': 'mean',
            'module_temp': 'mean'
        }).reset_index()

        if st.button("▶ Start Real-time Simulation"):
            chart_placeholder, metric_placeholder = st.empty(), st.empty()
            for i in range(1, len(df_rt)):
                current_df = df_rt.iloc[:i].copy()
                latest = df_rt.iloc[i-1]
                current_df['date'] = current_df['original_ts'].dt.date
                plot_df = []
                for date, group in current_df.groupby('date'):
                    plot_df.append(group)
                    spacer = pd.DataFrame([{'original_ts': group['original_ts'].max() + timedelta(seconds=1), 'ac_power': np.nan, 'dc_power': np.nan}])
                    plot_df.append(spacer)
                final_plot_df = pd.concat(plot_df)

                with metric_placeholder.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("AC Power (Avg)", f"{latest['ac_power']:.2f} kWh")
                    c2.metric("Irradiation (Avg)", f"{latest['irradiation']:.4f} W/m²")
                    c3.metric("Module Temp (Avg)", f"{latest['module_temp']:.2f} °C")

                fig_rt = go.Figure()
                fig_rt.add_trace(go.Scatter(x=final_plot_df['original_ts'], y=final_plot_df['ac_power'], name="AC Power", line=dict(color="#2ecc71"), connectgaps=False))
                fig_rt.add_trace(go.Scatter(x=final_plot_df['original_ts'], y=final_plot_df['dc_power'], name="DC Power", line=dict(color="#3498db", dash='dash'), connectgaps=False))
                fig_rt.update_layout(template="plotly_white", height=450, xaxis_title="Timestamp", yaxis_title="Power Level (kWh)")
                chart_placeholder.plotly_chart(fig_rt, use_container_width=True)
                time.sleep(0.01)
    else: st.warning("No real-time feed data found.")
