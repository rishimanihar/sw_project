import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Southwest Disruption Predictor",
    page_icon="✈️",
    layout="wide"
)

# Paths (Relative, matching your folder structure)
MODEL_PATH = os.path.join('data', 'xgb_airline_model.json')
DATA_PATH = os.path.join('data', 'final_model_predictions.csv')

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please run main.py first.")
        st.stop()
    
    model = XGBRegressor()
    model.load_model(MODEL_PATH)
    
    # 2. Load History Data
    if not os.path.exists(DATA_PATH):
        st.error(f"Data not found at {DATA_PATH}. Please run main.py first.")
        st.stop()
        
    df = pd.read_csv(DATA_PATH)
    
    # Fix: Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Fix: Ensure airport_code exists (if missing, extract from OHE or just handle gracefully)
    if 'airport_code' not in df.columns:
        st.warning("Column 'airport_code' missing in historical data. Charts may be limited.")
        # Fallback: Create dummy column if needed, or just use index if meaningful
        df['airport_code'] = 'Unknown'
        
    return model, df

try:
    model, df = load_resources()
except Exception as e:
    st.error(f"Critical Error loading resources: {e}")
    st.stop()

# ==========================================
# 3. SIDEBAR: USER INPUTS
# ==========================================
st.sidebar.header("🛠️ Scenario Settings")
st.sidebar.markdown("Adjust conditions to simulate a flight.")

# Flight Details
st.sidebar.subheader("Flight Info")
# Get list of airports from the historical data (or hardcode your 25 list)
airport_list = sorted(df['airport_code'].unique().astype(str)) if 'airport_code' in df.columns else ["DEN", "ATL", "MDW", "DAL"]
selected_airport = st.sidebar.selectbox("Origin Airport", airport_list)

selected_month = st.sidebar.select_slider("Month", options=range(1, 13), value=1)
selected_hour = st.sidebar.slider("Hour of Day (24h)", 0, 23, 14)
selected_day = st.sidebar.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x], index=2)

# Weather Conditions
st.sidebar.subheader("Weather Conditions")
temp_c = st.sidebar.slider("Temperature (°C)", -20, 45, 15)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0, 35, 5, help="> 15 m/s usually causes delays")
visibility = st.sidebar.slider("Visibility (meters)", 0, 16000, 10000, step=500)
precip = st.sidebar.number_input("Precipitation (mm/hr)", 0.0, 50.0, 0.0, step=0.5)

st.sidebar.markdown("---")
col1, col2, col3 = st.sidebar.columns(3)
is_snow = col1.checkbox("Snow", value=False)
is_thunder = col2.checkbox("Thunder", value=False)
is_fog = col3.checkbox("Fog", value=False)

# Derived Logic for "Rain" vs "Snow"
is_rain = 1 if (precip > 0 and not is_snow) else 0

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.title("🌩️ Southwest Airlines Disruption Predictor")
st.markdown(f"""
**Predicting the probability of operational chaos.** *Model: XGBoost Regressor | Data: 2022-2024 | Airports: 25 Major Hubs*
""")

tab_pred, tab_viz, tab_raw = st.tabs(["🔮 Prediction Engine", "📊 Historical Insights", "💾 Raw Data"])

# ------------------------------------------
# TAB 1: PREDICTION ENGINE
# ------------------------------------------
with tab_pred:
    col_kpi, col_result = st.columns([1, 2])
    
    with col_kpi:
        st.info(f"**Simulating:** Flight from **{selected_airport}**")
        st.write(f"📅 **Time:** {selected_hour}:00 (Month {selected_month})")
        st.write(f"🌡️ **Temp:** {temp_c}°C")
        st.write(f"🌬️ **Wind:** {wind_speed} m/s")
        
        predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            # --- A. BUILD INPUT DATAFRAME ---
            # We must match the EXACT columns the model was trained on.
            # Since the model expects lags (history), we assume the weather has been CONSTANT
            # for the last 12 hours (Simplification for "What-If" scenarios).
            
            # 1. Basic Features
            input_row = {
                'temp_c': temp_c,
                'dew_point_c': temp_c - 2, # Approximation
                'wind_speed_ms': wind_speed,
                'visibility_m': visibility,
                'ceiling_m': 30000 if not is_fog else 500, # Approximation
                'precip_depth_mm': precip,
                'is_fog': int(is_fog),
                'is_rain': int(is_rain),
                'is_snow': int(is_snow),
                'is_thunder': int(is_thunder),
                'hour': selected_hour,
                'month': selected_month,
                'day_of_week': selected_day,
                
                # 2. Interaction Features (From your winning script)
                'wind_x_snow': wind_speed * int(is_snow),
                'temp_x_rain': temp_c * int(is_rain),
                'vis_x_fog': visibility * int(is_fog),
            }

            # 3. Lag & Rolling Features (Simulated History)
            # We assume current conditions = past conditions for this demo
            lag_features = [
                'wind_speed_ms', 'visibility_m', 'precip_depth_mm', 'is_thunder', 'is_snow'
            ]
            for feat in lag_features:
                input_row[f'{feat}_lag1'] = input_row[feat]
                input_row[f'{feat}_lag3'] = input_row[feat]
            
            roll_features = ['wind_speed_ms', 'precip_depth_mm', 'is_snow', 'is_thunder']
            for feat in roll_features:
                input_row[f'{feat}_roll6'] = input_row[feat]
                input_row[f'{feat}_roll12'] = input_row[feat]

            # --- B. HANDLE ONE-HOT ENCODING ---
            # Convert dictionary to DataFrame
            df_input = pd.DataFrame([input_row])
            
            # Get the model's expected feature names
            model_features = model.get_booster().feature_names
            
            # Initialize missing columns (airport columns) to 0
            for col in model_features:
                if col not in df_input.columns:
                    df_input[col] = 0
            
            # Set the selected airport column to 1
            target_col = f"airport_{selected_airport}"
            if target_col in df_input.columns:
                df_input[target_col] = 1
            
            # Reorder to match model perfectly
            df_input = df_input[model_features]

            # --- C. PREDICT ---
            raw_pred = model.predict(df_input)[0]
            
            # Scale to 0-100 (Approximate based on training distribution)
            # Min/Max derived from typical PCA outputs (e.g. -3 to +10)
            # You can adjust these bounds if your outputs consistently hit 0 or 100.
            SCALER_MIN = -2.0
            SCALER_MAX = 8.0
            
            score = ((raw_pred - SCALER_MIN) / (SCALER_MAX - SCALER_MIN)) * 100
            score = np.clip(score, 0, 100)

            # --- D. VISUALIZE RESULT ---
            st.metric(label="Disruption Score (0-100)", value=f"{score:.1f}")
            
            # Speedometer Gauge
            fig_gauge = px.bar(
                x=[score], 
                y=["Score"], 
                orientation='h', 
                range_x=[0, 100],
                text=[f"{score:.1f}"],
                color=[score],
                color_continuous_scale=['green', 'yellow', 'red']
            )
            fig_gauge.update_layout(height=150, xaxis_title="Disruption Level", yaxis_title=None, showlegend=False)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Interpretation
            if score < 30:
                st.success("✅ **Smooth Operations:** Minimal delays expected.")
            elif score < 70:
                st.warning("⚠️ **Moderate Disruption:** Expect some delays and potential rerouting.")
            else:
                st.error("🚨 **SEVERE DISRUPTION:** High probability of cancellations and major delays.")

# ------------------------------------------
# TAB 2: HISTORICAL INSIGHTS
# ------------------------------------------
with tab_viz:
    st.header("Analysis of Historical Data (2022-2024)")
    
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.subheader("Average Disruption by Airport")
        if 'airport_code' in df.columns:
            avg_by_airport = df.groupby('airport_code')['Final_Score_0_100'].mean().sort_values()
            fig_bar = px.bar(avg_by_airport, orientation='h', color=avg_by_airport.values, color_continuous_scale='Reds')
            fig_bar.update_layout(showlegend=False, xaxis_title="Avg Disruption Score")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Airport data missing from prediction file.")

    with row1_col2:
        st.subheader("Disruption by Hour of Day")
        if 'timestamp' in df.columns:
            df['plot_hour'] = df['timestamp'].dt.hour
            avg_by_hour = df.groupby('plot_hour')['Final_Score_0_100'].mean()
            fig_line = px.line(avg_by_hour, markers=True)
            fig_line.update_layout(xaxis_title="Hour of Day", yaxis_title="Avg Disruption Score")
            st.plotly_chart(fig_line, use_container_width=True)

# ------------------------------------------
# TAB 3: RAW DATA
# ------------------------------------------
with tab_raw:
    st.subheader("Model Predictions Sample")
    st.dataframe(df.head(1000))