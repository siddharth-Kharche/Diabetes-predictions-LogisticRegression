from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app=application

scaler=pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")


#streamlit
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
