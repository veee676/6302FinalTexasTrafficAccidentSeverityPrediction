import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap


from utils.feature_engineering import process_input
from rag.rag_pipeline import generate_explanation

# loading models
clf = joblib.load('models/clf_model.pkl')
reg = joblib.load('models/regression_model.pkl')

st.title("Crash Severity & Injury Prediction")
st.header("Input Parameters")

crash_speed_limit = st.slider("Speed Limit", 0, 100, 45)
num_units = st.slider("Number of Units involved (vehicle or otherwise)", 1, 10, 2)
hour = st.slider("Hour of Day", 0, 23, 12)

collision_type = st.selectbox("Collision Type", 
                              ['head_on', 'parking_related', 'sideswipe', 'single_vehicle', 'turning'])

explainer = shap.TreeExplainer(clf)

if st.button('Predict'):

    input_data = {
        'crash_speed_limit': crash_speed_limit,
        'num_units': num_units,
        'hour': hour,
        'collision_type': collision_type
    }

    df =process_input(input_data)

    severity_pred = clf.predict(df)[0]
    injury_pred = reg.predict(df)[0]

    explanation = generate_explanation(
        input_data, severity_pred, injury_pred)
    
    st.subheader("Results")
    st.write(f"Severity: {severity_pred}")
    st.write(f"Predicted Injuries: {injury_pred:0.2f}")

    st.subheader("Explanation")
    st.write(explanation)

    shap_vals = explainer.shap_values(df)

    st.subheader("Feature Importance (SHAP)")

    fig, ax = plt.subplot()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_vals[1][0],
            base_vals=explainer.expected_value[1],
            data=df.iloc[0],
            feature_names=df.columns
        ),
        show=False
    )

    st.pyplot(fig)