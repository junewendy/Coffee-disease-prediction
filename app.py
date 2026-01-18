import streamlit as st
import pandas as pd
import joblib
# These imports are CRITICAL to prevent the AttributeError
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="Coffee Disease Risk AI", page_icon="☕")
st.title("☕ Coffee Disease Risk Prediction")
st.markdown("Enter the farm details below to assess the risk of Coffee Leaf Rust.")

# --- 2. LOAD THE MODEL ---
@st.cache_resource
def load_my_model():
    # Since you saved 'final_pipeline', this one file contains everything!
    return joblib.load('coffee_disease_model_v1.pkl')

model = load_my_model()

# --- 3. USER INPUTS ---
st.sidebar.header("Input Weather & Farm Data")

# Numeric Inputs matching your X columns
temp = st.sidebar.number_input("Average Temperature (°C)", value=22.0)
hum = st.sidebar.number_input("Humidity (%)", value=70.0)
rain = st.sidebar.number_input("Rainfall (mm)", value=5.0)
wind = st.sidebar.number_input("Wind Speed (m/s)", value=2.0)
month = st.sidebar.slider("Month (1-12)", 1, 12, 6)

# Crop Stage matching your OrdinalEncoder
stage = st.sidebar.selectbox("Crop Stage", 
                            options=["Flowering", "Berry Development", "Ripening", "Harvesting"])

# --- 4. PREPARE DATA FOR PREDICTION ---
if st.button("Analyze Risk Level"):
    # We must create a DataFrame with the EXACT names and order as your X index
    input_df = pd.DataFrame({
        'Temp(Avg)': [temp],
        'Humidity(%)': [hum],
        'Rainfall(mm)': [rain],
        'WindSpeed(m/s)': [wind],
        'CropStage': [stage],
        'Month': [month],
        'Temp(Avg)_Lag14': [temp], # Using current values as proxies for lags
        'Humidity(%)_Lag14': [hum],
        'Rainfall(mm)_Lag14': [rain],
        'WindSpeed(m/s)_Lag14': [wind],
        'Temp(Avg)_Avg_Last14Days': [temp],
        'Humidity(%)_Avg_Last14Days': [hum],
        'Rainfall(mm)_Avg_Last14Days': [rain],
        'WindSpeed(m/s)_Avg_Last14Days': [wind]
    })

    # Ensure column order matches exactly what the pipeline expects
    # (Based on your X = df_final.drop(columns=['Date', 'RiskLabel(Target)']))
    
    # Get Prediction
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    confidence = max(prob) * 100

    # --- 5. DISPLAY RESULTS ---
    # Assuming 0=Low, 1=Medium, 2=High based on your previous description
    if prediction == 2:
        st.error(f"🔴 HIGH RISK DETECTED ({confidence:.1f}% Confidence)")
        st.write("❗ **Action:** Immediate monitoring and preventive spraying recommended.")
    elif prediction == 1:
        st.warning(f"🟡 MEDIUM RISK DETECTED ({confidence:.1f}% Confidence)")
        st.write("⚠️ **Caution:** Conditions are favorable for disease. Increase scouting.")
    else:
        st.success(f"🟢 LOW RISK ({confidence:.1f}% Confidence)")
        st.write("✅ **Status:** Conditions are currently stable.")