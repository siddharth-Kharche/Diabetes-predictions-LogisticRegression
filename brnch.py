streamlit
import streamlit as st
import pickle
import numpy as np

# Load the scaler and model
scaler = pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

# Title and description
st.title('Diabetes Prediction')
st.write("Enter the details below to check if the patient is diabetic or non-diabetic.")

# Input fields for user data
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
Glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0, step=0.1)
BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, step=0.1)
SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, step=0.1)
Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, step=0.1)
BMI = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
Age = st.number_input("Age", min_value=0, max_value=120, step=1)

# When the "Predict" button is pressed
if st.button("Predict"):
    # Collect the data into a numpy array and scale it
    new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Make the prediction
    prediction = model.predict(new_data)
    
    # Output the result
    if prediction[0] == 1:
        st.success('The patient is Diabetic')
    else:
        st.success('The patient is Non-Diabetic')
