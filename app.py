import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_model (1).json")
    return model


# --------------------------------------------------
# LOAD PREPROCESSOR
# --------------------------------------------------

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor (1).pkl")


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

try:
    model = load_model()
    preprocessor = load_preprocessor()

except Exception as e:
    st.error("Error loading model or preprocessor.")
    st.code(str(e))
    st.stop()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("👨‍💼 Employee Attrition Prediction System")

st.write(
    "Enter employee information below to predict the probability of employee attrition."
)

st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.header("Employee Information")


col1, col2, col3 = st.columns(3)


# ---------------- NUMERICAL INPUTS ----------------

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30
    )

    years_company = st.number_input(
        "Years at Company",
        min_value=0,
        max_value=50,
        value=5
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=5000
    )

    promotions = st.number_input(
        "Number of Promotions",
        min_value=0,
        max_value=20,
        value=1
    )

    distance_home = st.number_input(
        "Distance from Home",
        min_value=0,
        max_value=100,
        value=10
    )


with col2:

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=10,
        value=1
    )

    company_tenure = st.number_input(
        "Company Tenure",
        min_value=0,
        max_value=50,
        value=5
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Technology",
            "Finance",
            "Healthcare",
            "Media"
        ]
    )

    work_life_balance = st.selectbox(
        "Work Life Balance",
        [
            "Good",
            "High",
            "Average"
        ]
    )


with col3:

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [
            "High",
            "Average",
            "Low",
            "Very High"
        ]
    )

    performance_rating = st.selectbox(
        "Performance Rating",
        [
            "Excellent",
            "Good",
            "Average",
            "Fair",
            "Poor"
        ]
    )

    overtime = st.selectbox(
        "Overtime",
        [
            "Yes",
            "No"
        ]
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "High School",
            "Associate Degree",
            "Bachelor's Degree",
            "Master",
            "PhD"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


st.divider()

st.header("Additional Employee Information")


col4, col5, col6 = st.columns(3)


with col4:

    job_level = st.selectbox(
        "Job Level",
        [
            "Entry",
            "Mid",
            "Senior"
        ]
    )

    company_size = st.selectbox(
        "Company Size",
        [
            "Small",
            "Medium",
            "Large"
        ]
    )


with col5:

    remote_work = st.selectbox(
        "Remote Work",
        [
            "Yes",
            "No"
        ]
    )

    leadership_opportunities = st.selectbox(
        "Leadership Opportunities",
        [
            "Yes",
            "No"
        ]
    )


with col6:

    innovation_opportunities = st.selectbox(
        "Innovation Opportunities",
        [
            "Yes",
            "No"
        ]
    )

    company_reputation = st.selectbox(
        "Company Reputation",
        [
            "Excellent",
            "Good",
            "Average",
            "Fair",
            "Poor"
        ]
    )

    employee_recognition = st.selectbox(
        "Employee Recognition",
        [
            "Excellent",
            "Good",
            "Average",
            "Fair",
            "Poor"
        ]
    )


st.divider()


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button(
    "🔮 Predict Employee Attrition",
    use_container_width=True
):

    # ----------------------------------------------
    # CREATE INPUT DATAFRAME
    # ----------------------------------------------

    input_data = pd.DataFrame({

        "Age": [age],

        "Years at Company": [years_company],

        "Monthly Income": [monthly_income],

        "Number of Promotions": [promotions],

        "Distance from Home": [distance_home],

        "Number of Dependents": [dependents],

        "Company Tenure": [company_tenure],

        "Gender": [gender],

        "Job Role": [job_role],

        "Work Life Balance": [work_life_balance],

        "Job Satisfaction": [job_satisfaction],

        "Performance Rating": [performance_rating],

        "Overtime": [overtime],

        "Education Level": [education_level],

        "Marital Status": [marital_status],

        "Job Level": [job_level],

        "Company Size": [company_size],

        "Remote Work": [remote_work],

        "Leadership Opportunities": [leadership_opportunities],

        "Innovation Opportunities": [innovation_opportunities],

        "Company Reputation": [company_reputation],

        "Employee Recognition": [employee_recognition],"Work-Life Balance": [work_life_balance],
    })


    # ----------------------------------------------
    # PREPROCESS INPUT
    # ----------------------------------------------

    try:

        input_transformed = preprocessor.transform(input_data)

    except Exception as e:

        st.error("Error while preprocessing the input data.")

        st.code(str(e))

        st.stop()


    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    try:

        prediction_probability = model.predict_proba(
            input_transformed
        )[0][1]

        prediction = model.predict(
            input_transformed
        )[0]

    except Exception as e:

        st.error("Error while making prediction.")

        st.code(str(e))

        st.stop()


    # ----------------------------------------------
    # DISPLAY RESULT
    # ----------------------------------------------

    st.subheader("Prediction Result")

    probability = prediction_probability * 100


    if prediction == 1:

        st.error(
            f"⚠️ Employee Attrition Predicted"
        )

    else:

        st.success(
            f"✅ Employee Not Likely to Leave"
        )


    st.metric(
        "Attrition Probability",
        f"{probability:.2f}%"
    )


    # ----------------------------------------------
    # PROGRESS BAR
    # ----------------------------------------------

    st.progress(
        float(min(max(prediction_probability, 0.0), 1.0))
    )


    # ----------------------------------------------
    # SHOW INPUT DATA
    # ----------------------------------------------

    with st.expander("View Employee Information"):

        st.dataframe(
            input_data,
            use_container_width=True
        )