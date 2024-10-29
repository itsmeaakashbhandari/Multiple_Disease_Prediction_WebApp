# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:14:23 2024

@author: Asus
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size: 28px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subheader {
            font-size: 18px;
            font-weight: bold;
            color: #2196F3;
        }
        .sidebar .sidebar-content {
            background-color: #F8F9FA;
        }
        .stButton > button {
            color: white;
            background-color: #FF4B4B;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    
    st.title("Multiple Disease Prediction System")
    selected = option_menu(
        "Choose a Prediction Model",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
        icons=["activity", "heart", "person"],
        menu_icon="stethoscope",
        default_index=0
    )







# Load saved models
diabetes_model = pickle.load(open('trained_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# Helper function for displaying results
def show_result(diagnosis):
    if diagnosis:
        st.success(diagnosis, icon="✅")
    else:
        st.warning("Please enter all required values.", icon="⚠️")

# Diabetes Prediction Page
if selected == "Diabetes Prediction":
    st.markdown('<div class="title">Diabetes Prediction using ML</div>', unsafe_allow_html=True)

    # Collect input data
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1, step=1)
    Glucose = st.slider("Glucose Level", min_value=0, max_value=200, value=120)
    BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=180, value=70)
    SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    Insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=30)
    BMI = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    Age = st.slider("Age of the Person", min_value=0, max_value=120, value=30)

    # Prediction and result display
    if st.button("Diabetes Test Result"):
        prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diagnosis = "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
        show_result(diagnosis)

# Heart Disease Prediction Page
if selected == "Heart Disease Prediction":
    st.markdown('<div class="title">Heart Disease Prediction using ML</div>', unsafe_allow_html=True)

    # Collect input data
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.slider("Chest Pain Type", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.slider("Resting ECG Result", min_value=0, max_value=2, value=1)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.radio("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=5.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.slider("Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # Convert categorical inputs to numeric
    sex = 1 if sex == "Male" else 0

    # Prediction and result display
    if st.button("Heart Disease Test Result"):
        try:
            # Ensure all inputs are converted to the appropriate numeric types
            heart_prediction = heart_disease_model.predict([[float(age), float(sex), float(cp), float(trestbps), float(chol), 
                                                             float(fbs), float(restecg), float(thalach), float(exang), 
                                                             float(oldpeak), float(slope), float(ca), float(thal)]])
            diagnosis = "The person has heart disease" if heart_prediction[0] == 1 else "The person does not have heart disease"
            show_result(diagnosis)
        except ValueError:
            st.error("Please make sure all inputs are filled correctly and are in numeric format.")


# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    st.markdown('<div class="title">Parkinson\'s Disease Prediction using ML</div>', unsafe_allow_html=True)

    # Collect input data
    fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, max_value=300.0, value=120.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, max_value=500.0, value=200.0)
    flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, max_value=300.0, value=90.0)
    Jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.1, value=0.005, step=0.001)
    Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.01, value=0.0001, step=0.0001)
    RAP = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.1, value=0.003, step=0.001)
    PPQ = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.1, value=0.005, step=0.001)
    DDP = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0, value=0.1)
    Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=10.0, value=1.0)
    APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=1.0, value=0.01)
    APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=1.0, value=0.02)
    APQ = st.number_input("MDVP:APQ", min_value=0.0, max_value=1.0, value=0.03)
    DDA = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.1, value=0.04)
    NHR = st.number_input("NHR", min_value=0.0, max_value=1.0, value=0.1)
    HNR = st.number_input("HNR", min_value=0.0, max_value=40.0, value=20.0)
    RPDE = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.5)
    DFA = st.number_input("DFA", min_value=0.0, max_value=2.0, value=0.8)
    spread1 = st.number_input("spread1", min_value=-10.0, max_value=10.0, value=-5.0)
    spread2 = st.number_input("spread2", min_value=-10.0, max_value=10.0, value=3.0)
    D2 = st.number_input("D2", min_value=0.0, max_value=5.0, value=1.0)
    PPE = st.number_input("PPE", min_value=0.0, max_value=1.0, value=0.5)

    # Prediction and result display
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                                                           RPDE, DFA, spread1, spread2, D2, PPE]])
        diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
        show_result(diagnosis)

