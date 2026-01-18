import streamlit as st
import pandas as pd
import joblib

# 1. Load the model
model = joblib.load('coffee_disease_model_v1.pkl')

st.set_page_config(page_title="Coffee Risk AI", page_icon="☕")
st.title("☕ Coffee Leaf Rust: Early Warning System")
st.markdown("---")

# 2. User Inputs
st.sidebar.header("Input Weather Conditions")
temp = st.sidebar.number_input("Current Temp (Avg) °C", value=22.0)
hum = st.sidebar.slider("Current Humidity (%)", 0, 100, 60)
rain = st.sidebar.number_input("Current Rainfall (mm)", value=5.0)
wind = st.sidebar.number_input("Wind Speed (m/s)", value=2.0)
# Optional: Add a selector for CropStage so it's not hardcoded
stage = st.sidebar.selectbox("Crop Stage", ['Flowering', 'Fruit Setting', 'Harvesting'])

# 3. Analyze Button
if st.button("Analyze Risk Level"):
    # Create the DataFrame with EXACT matches to your training column names
    data = {
        'Temp(Avg)': [temp], 
        'Humidity(%)': [hum], 
        'Rainfall(mm)': [rain], 
        'WindSpeed(m/s)': [wind],
        'Month': [1], 
        'CropStage': [stage],
        'Temp(Avg)_Lag14': [temp], 
        'Humidity(%)_Lag14': [hum], 
        'Rainfall(mm)_Lag14': [rain], 
        'WindSpeed(m/s)_Lag14': [wind], # FIXED: Removed space
        'Temp(Avg)_Avg_Last14Days': [temp], 
        'Humidity(%)_Avg_Last14Days': [hum], 
        'Rainfall(mm)_Avg_Last14Days': [rain], 
        'WindSpeed(m/s)_Avg_Last14Days': [wind] # FIXED: Removed space
    }
    
    input_df = pd.DataFrame(data)
    
    # CRITICAL: Force the column order to match what the model saw during training
    cols_order = [
        'Temp(Avg)', 'Humidity(%)', 'Rainfall(mm)', 'WindSpeed(m/s)',
        'CropStage', 'Month', 'Temp(Avg)_Lag14', 'Humidity(%)_Lag14', 
        'Rainfall(mm)_Lag14', 'WindSpeed(m/s)_Lag14', 'Temp(Avg)_Avg_Last14Days', 
        'Humidity(%)_Avg_Last14Days', 'Rainfall(mm)_Avg_Last14Days', 'WindSpeed(m/s)_Avg_Last14Days'
    ]
    input_df = input_df[cols_order]
    
    # 4. Prediction & Mapping
    prediction_numeric = model.predict(input_df)[0]
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    prediction_text = risk_map.get(prediction_numeric, str(prediction_numeric))
    
    # 5. The Professional Output
    if prediction_text == 'High':
        st.error(f"### 🚨 Predicted Risk: {prediction_text}")
        st.warning("**Recommendation:** Initiate preventive fungicide application immediately and alert regional officers.")
    elif prediction_text == 'Medium':
        st.warning(f"### 🔔 Predicted Risk: {prediction_text}")
        st.info("**Recommendation:** Increase monitoring frequency. Inspect lower leaf surfaces for yellow-orange spores.")
    else:
        st.success(f"### ✅ Predicted Risk: {prediction_text}")
        st.write("**Recommendation:** No immediate action required. Continue standard maintenance.")