import streamlit as st
import pandas as pd
import numpy as np
import pymongo
import plotly.express as px
import plotly.graph_objects as go
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

# ====================== TAB 1: PERFORMANCE METRICS ======================
with tabs[0]:
    st.subheader("📊 KPI & Yield Analysis")
    
    if not filtered_perf.empty:
        total_act = filtered_perf['metrics.actual_daily_yield'].sum()
        total_pre = filtered_perf['metrics.predicted_daily_yield'].sum()
        gap = total_act - total_pre

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

        # Biểu đồ kết hợp theo Inverter (Giữ nguyên thứ tự)
        st.subheader("⚖️ So sánh Sản lượng theo từng Inverter")

        df_inv = filtered_perf.groupby('source_key').agg({
            'metrics.actual_daily_yield': 'sum',
            'metrics.predicted_daily_yield': 'sum'
        }).reset_index()

        # KHÔNG sắp xếp lại để giống với ngrok
        fig_combined = go.Figure()

        fig_combined.add_trace(go.Bar(
            x=df_inv['source_key'],
            y=df_inv['metrics.actual_daily_yield'],
            name='Actual Yield (Thực tế)',
            marker_color='#3498db',
            text=df_inv['metrics.actual_daily_yield'].round(1),
            textposition='auto'
        ))

        fig_combined.add_trace(go.Scatter(
            x=df_inv['source_key'],
            y=df_inv['metrics.predicted_daily_yield'],
            name='AI Predicted Target',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3.5),
            marker=dict(size=9)
        ))

        fig_combined.update_layout(
            title="So sánh Sản lượng Thực tế vs AI Dự báo theo từng Inverter",
            template="plotly_white",
            height=520,
            xaxis_title="Inverter ID (source_key)",
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
        if sim_weather == "Cloudy":
            expected_yield *= 0.7
        elif sim_weather == "Partly Cloudy":
            expected_yield *= 0.9

    with sc2:
        fig_sim = go.Figure(go.Indicator(
            mode="gauge+number", 
            value=expected_yield, 
            title={'text': "Predicted Power (kW)"},
            gauge={'axis': {'range': [None, 2000]}, 'bar': {'color': "#3498db"}}
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
    
    if not df_alerts.empty:
        st.table(df_alerts[['source_key', 'severity', 'root_cause_analysis.suggested_action']].head(10))

# ====================== TAB 4: Deep Analytics ======================
with tabs[3]:
    st.subheader("Advanced Predictive Analytics")
    if not filtered_weather.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_ts = px.area(
                filtered_weather.sort_values('timestamp'), 
                x="timestamp", 
                y="predictions.expected_ac_power", 
                title="Projected Generation Potential"
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        with col2:
            fig_hm = px.density_heatmap(
                filtered_weather, 
                x="features_snapshot.module_temp", 
                y="features_snapshot.irradiation", 
                z="predictions.expected_ac_power", 
                histfunc="avg", 
                title="Density Heatmap", 
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_hm, use_container_width=True)

# ====================== TAB 5: Real-time Monitoring ======================
with tabs[4]:
    st.subheader("⚡ Live Data Stream Simulator")
    
    rt_cursor = db['realtime_feeds'].find().sort("original_ts", 1).allow_disk_use(True)
    df_rt_raw = pd.DataFrame(list(rt_cursor))
    
    if df_rt_raw.empty:
        st.warning("No real-time feed data found in MongoDB.")
    else:
        scaling_features = ['dc_power', 'ac_power', 'irradiation']
        for feature in scaling_features:
            if feature == 'dc_power': idx = 0
            elif feature == 'ac_power': idx = 1
            elif feature == 'irradiation': idx = 2
            else: continue
                
            min_val = scaler_minmax.min_[idx]
            scale_val = scaler_minmax.scale_[idx]
            df_rt_raw[feature] = np.expm1((df_rt_raw[feature] - min_val) / scale_val)
        
        if 'module_temp' in df_rt_raw.columns:
            t_idx = 0
            df_rt_raw['module_temp'] = (df_rt_raw['module_temp'] * scaler_standard.scale_[t_idx]) + scaler_standard.mean_[t_idx]
        
        df_rt = df_rt_raw.groupby('original_ts').agg({
            'ac_power': 'mean',
            'dc_power': 'mean',
            'irradiation': 'mean',
            'module_temp': 'mean'
        }).reset_index()
        
        if st.button("▶ Start Real-time Simulation"):
            chart_placeholder = st.empty()
            metric_placeholder = st.empty()
            
            for i in range(1, len(df_rt) + 1):
                current_df = df_rt.iloc[:i].copy()
                latest = df_rt.iloc[i-1]
                
                with metric_placeholder.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("AC Power (Avg)", f"{latest['ac_power']:.2f} kW")
                    c2.metric("Irradiation (Avg)", f"{latest['irradiation']:.2f} W/m²")
                    c3.metric("Module Temp (Avg)", f"{latest['module_temp']:.1f} °C")
                
                fig_rt = go.Figure()
                fig_rt.add_trace(go.Scatter(x=current_df['original_ts'], y=current_df['ac_power'], 
                                          name="AC Power", line=dict(color="#2ecc71")))
                fig_rt.add_trace(go.Scatter(x=current_df['original_ts'], y=current_df['dc_power'], 
                                          name="DC Power", line=dict(color="#3498db", dash='dash')))
                
                fig_rt.update_layout(
                    template="plotly_white", 
                    height=450, 
                    xaxis_title="Timestamp", 
                    yaxis_title="Power (kW)"
                )
                chart_placeholder.plotly_chart(fig_rt, use_container_width=True)
                time.sleep(0.08)
