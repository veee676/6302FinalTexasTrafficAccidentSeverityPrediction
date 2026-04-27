import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap


from utils.feature_engineering import process_input
from rag.rag_pipeline import generate_explanation

# loading models
clf = joblib.load('models/clf_agent.pkl')
reg = joblib.load('models/reg_agent.pkl')

clf_explainer = shap.TreeExplainer(clf)
reg_explainer = shap.TreeExplainer(reg)


st.title("Crash Severity & Injury Prediction")
st.header("Input Parameters")

# BASIC INPUTS
crash_speed_limit = st.slider("Speed Limit", 0, 100, 35)
num_units = st.slider("Number of Vehicles", 1, 10, 1)
hour = st.slider("Hour of Day", 0, 23, 12)

# DAY OF WEEK
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

day_label = st.selectbox("Day of Week", list(day_map.keys()))
day_of_week = day_map[day_label]

# COLLISION TYPE
collision_type = st.selectbox(
    "Collision Type",
    [
        "Head-on",
        "Rear-end",
        "Sideswipe",
        "Single vehicle",
        "Turning",
        "Parking",
        "Other"
    ]
)

# PREDICT
if st.button("Predict"):

    input_data = {
        "crash_speed_limit": crash_speed_limit,
        "num_units": num_units,
        "hour": hour,
        "day_of_week": day_of_week,
        "collision_type": collision_type
    }

    df = process_input(input_data)

    # Debug check 
    # st.write(df)

    severity_pred = clf.predict(df)[0]
    injury_pred = reg.predict(df)[0]

    explanation = generate_explanation(
        input_data, severity_pred, injury_pred)
    
    st.subheader("Results")
    st.write(f"Severity: {severity_pred}")
    st.write(f"Predicted Injuries: {injury_pred:0.2f}")

    st.subheader("Explanation")
    st.write(explanation)

    st.subheader("Feature Impact on Severity (SHAP)")
    

    shap_vals = clf_explainer.shap_values(df)
    st.write("SHAP Shape:", getattr(shap_vals, 'shape', 'list'))

    if isinstance(shap_vals, list):
        # old shap frmt
        pred_class = int(severity_pred)
        pred_class = min(pred_class, len(shap_vals) - 1)

        shap_val = shap_vals[pred_class][0]
        base_val = clf_explainer.expected_value[pred_class]
    else:
        # new shap frmt
        pred_class = int(severity_pred)
        shap_val = shap_vals[0, :, pred_class]
        base_val = clf_explainer.expected_value[pred_class]

    fig1, ax1 = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_val,
            base_values=base_val,
            data=df.iloc[0],
            feature_names=df.columns.tolist()
        ),
        show=False
    )

    st.pyplot(fig1)

    st.subheader("Feature Impact on Injury Prediction (SHAP)")
    st.write("SHAP Shape:", getattr(shap_vals, 'shape', 'list'))


    shap_vals = reg_explainer.shap_values(df)

    shap_val = shap_vals[0]   # first sample
    base_val = reg_explainer.expected_value

    fig2, ax2 = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_val,
            base_values=base_val,
            data=df.iloc[0],
            feature_names=df.columns.tolist()
        ),
        show=False
    )

    st.pyplot(fig2)