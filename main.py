import streamlit as st
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import PowerTransformer



#Load model and transformer
model = joblib.load("cellphone_price_prediction_model.pkl")
pt = joblib.load("pt.pkl")



st.title("📱 Mobile Price Prediction App")

st.write("Enter the specifications below to predict the price:")

# Input fields
sale = st.number_input("Sale", min_value=10.0, step=10.0)
weight = st.number_input("Weight (grams)", min_value=66.0, max_value = 800.0, step=1.0)
resolution = st.number_input("Resolution", min_value=1.0,max_value = 20.0, step=0.1)
ppi = st.number_input("PPI", min_value=1.0, max_value = 800.0,step=1.0)
cpu_core = st.number_input("CPU Cores", min_value=1.0,max_value = 20.0, step=1.0)
cpu_freq = st.number_input("CPU Frequency (GHz)", min_value=0.0, step=0.1)
internal_mem = st.number_input("Internal Memory (GB)", min_value=0.0, step=1.0)
ram = st.number_input("RAM (GB)", min_value=0.0, step=0.1)
rear_cam = st.number_input("Rear Camera (MP)", min_value=0.0, step=0.1)
front_cam = st.number_input("Front Camera (MP)", min_value=0.0, step=0.1)
battery = st.number_input("Battery (mAh)", min_value=500.0, step=100.0)
thickness = st.number_input("Thickness (mm)", min_value=0.0, step=0.1)

features = np.array([[sale, weight, resolution, ppi, cpu_core, cpu_freq,
                      internal_mem, ram, rear_cam, front_cam, battery, thickness]])

features = pt.transform(features)



# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(features)
    st.success(f"💰 Predicted Price: {prediction[0]:.2f}")
