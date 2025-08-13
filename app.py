import streamlit as st
import numpy as np
import pickle
from keras.models import load_model

# Load trained model and scaler
model = load_model('student_perfomance.h5')  # ✅ This is your ANN model
scaler = pickle.load(open('student_perfomance.pkl', 'rb'))  # ✅ This is just the StandardScaler

# Streamlit page config
st.set_page_config(page_title="Student Performance Predictor", page_icon="", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #3b82f6;'>Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #2563eb;'>With Great Respect to <i>Sir Hamza & Sir Shahzeb</i></h3>", unsafe_allow_html=True)
st.markdown("---")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Student Details:")
    
    school = st.selectbox("School", ["Allied School", "The Smart School"])
    sex = st.selectbox("Gender", ["M", "F"])
    age = st.number_input("Age", min_value=10, max_value=30, value=18)
    studytime = st.slider("Weekly Study Time (1-4)", 1, 4, 2)
    failures = st.slider("Past Class Failures", 0, 4, 0)
    G1 = st.number_input("G1 (First Period Grade)", 0, 20, 10)
    G2 = st.number_input("G2 (Second Period Grade)", 0, 20, 10)

    submit = st.form_submit_button("Predict Final Grade")

# Preprocess and predict
if submit:
    # Encode categorical inputs
    school = 0 if school == "Allied School" else 1
    sex = 0 if sex == "F" else 1

    # Combine inputs into array
    input_data = np.array([[school, sex, age, studytime, failures, G1, G2]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict using the ANN model
    prediction = model.predict(input_scaled)
    grade = round(prediction[0][0], 2)

if grade >= 16:
    remark = "Excellent!"
elif grade >= 12:
    remark = "Good"
elif grade >= 10:
    remark = "Passed"
else:
    remark = "Needs Improvement"

  # Display result
st.success(f" Predicted Final Grade (G3): {grade} / 20 — {remark}")

# Footer
st.markdown("---")
st.markdown("<footer style='text-align: center;'> Developed by <strong>Bushra Sarwar</strong></footer>", unsafe_allow_html=True)
