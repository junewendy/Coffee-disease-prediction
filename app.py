import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# --- 1. SETUP PAGE ---
st.set_page_config(page_title="Coffee Disease Risk AI", page_icon="☕")
st.title("☕ Coffee Disease Risk Prediction")
st.markdown("Enter the weather and farm details below to assess the risk of Coffee Leaf Rust.")

# --- 2. DEFINE PREPROCESSOR (REQUIRED TO FIX ATTRIBUTEERROR) ---
# IMPORTANT: These names must match your training data exactly!
numeric_features = [
    'Temp(Avg)', 'Humidity(%)', 'Rainfall(mm)', 'WindSpeed(m/s)', 
    'Temp(Avg)_Lag14', 'Humidity(%)_Lag14', 'Rainfall(mm)_Lag14', 'WindSpeed(m/s)_Lag14'
]
categorical_features = ['CropStage']

# Defining the structure helps joblib understand how to "unpickle" the model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OrdinalEncoder(), categorical_features)
    ])

# --- 3. LOAD THE MODEL ---
@st.cache_resource
def load_my_model():
    # This loads your saved Random Forest Pipeline
    return joblib.load('coffee_disease_model_v1.pkl')

model = load_my_model()

# --- 4. USER INPUTS (SIDEBAR OR MAIN) ---
st.sidebar.header("Current Weather Conditions")
temp = st.sidebar.number_input("Average Temperature (°C)", value=22.0)
hum = st.sidebar.number_input("Humidity (%)", value=70.0)
rain = st.sidebar.number_input("Rainfall (mm)", value=5.0)
wind = st.sidebar.number_input("Wind Speed (m/s)", value=2.0)
stage = st.sidebar.selectbox("Crop Stage", options=["Flowering", "Berry Development", "Ripening", "Harvesting"])

# --- 5. MAKE PREDICTION ---
if st.button("Analyze Risk Level"):
    # Create a dataframe for the input (using the features the model expects)
    input_data = pd.DataFrame({
        'Temp(Avg)': [temp],
        'Humidity(%)': [hum],
        'Rainfall(mm)': [rain],
        'WindSpeed(m/s)': [wind],
        'Temp(Avg)_Lag14': [temp],  # Using current as proxy for lag for demo
        'Humidity(%)_Lag14': [hum],
        'Rainfall(mm)_Lag14': [rain],
        'WindSpeed(m/s)_Lag14': [wind],
        'CropStage': [stage]
    })
    
   # Get Prediction
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    confidence = max(prob) * 100

    # --- UPDATED LOGIC FOR 3 CLASSES ---
    if prediction == 2:  # High Risk
        st.error(f"🔴 HIGH RISK DETECTED ({confidence:.1f}% Confidence)")
        st.write("❗ **Action Required:** Immediate monitoring and preventive spraying recommended.")
        
    elif prediction == 1:  # Medium Risk
        st.warning(f"🟡 MEDIUM RISK DETECTED ({confidence:.1f}% Confidence)")
        st.write("⚠️ **Caution:** Weather conditions are becoming favorable for disease. Monitor closely.")
        
    else:  # Low Risk (prediction == 0)
        st.success(f"🟢 LOW RISK ({confidence:.1f}% Confidence)")
        st.write("✅ **Status:** Conditions are currently stable. Continue routine farm maintenance.")