
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="SPAT - Student Performance",
    page_icon="🎓",
    layout="wide"
)

# Load trained model
saved_data = joblib.load("KNN_model.pkl")

model = saved_data["model"]
scalar = saved_data["Scalar"]

st.title("🎓 SPAT - Student Performance Prediction")
st.subheader("Intelligent Student Performance Analysis")

st.divider()

st.header("Student Information")

col1, col2, col3 = st.columns(3)

with col1:
    hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

with col3:
    prevmarks = st.number_input(
        "Previous Marks",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

st.divider()

if st.button("Predict Student Result", type="primary"):

    input_data = pd.DataFrame(
        [[hours, attendance, prevmarks]],
        columns=["Hours", "Attendance", "PrevMarks"]
    )

    scaled_data = scalar.transform(input_data)

    prediction = model.predict(scaled_data)

    st.subheader("Prediction Result")

    if prediction[0] == "pass":
        st.success("Congratulations! Student is predicted to PASS 🎉")
    else:
        st.error("Student is predicted to FAIL.")

st.divider()

st.caption("SPAT | Student Performance Analysis Tool")
