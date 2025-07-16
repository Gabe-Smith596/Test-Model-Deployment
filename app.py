#Importing Libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib

#Loading the saved components
model = joblib.load('gradient_boosting_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')


st.title("Glass Type Prediction")
st.write("This model predicts the glass type according to the input features below. Please enter your values: ")

#The features outlined here should match the features used during the training of the model

#Input features

RI = st.slider("Refractive Index", 1.5, 1.8)
Na = st.slider("Sodium Content", 10.0, 18.0)
Mg = st.slider("Magnesium Content", 0.0, 4.0)
Al = st.slider("Aluminium Content", 0.0, 4.0)
Si = st.slider("Silicon Content", 70.0, 80.0)
K = st.slider("Potassium Content", 0.0, 0.5)
Ca = st.slider("Calcium Content", 5.0, 10.0)
Ba = st.slider("Barium Content", 0.0, 5.0)
Fe = st.slider("Iron Content", 0.0, 5.0)

#Preparing Input Features for Model
features = np.array([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]])
scaled_features = scaler.transform(features)


#Prediction
if st.button("Predict Glass Type"):
    prediction_encoded = model.predict(scaled_features)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

    st.success(f"Predicted Glass Type: {prediction_label}")
