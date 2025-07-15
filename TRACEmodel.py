import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import joblib
import gdown
import os
import tempfile

@st.cache_resource 
def load_model():
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, "rsfmodel.sav")
    
    if not os.path.exists(model_path):
        st.info("Loading model...")
        file_id = "1BUwxs50NOH3yMjhmgCvarcc4vMkjSHFa"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            st.stop()
    
    return joblib.load(model_path)

rsf = load_model()

st.title('Prediction model for SLD-HCC (STEALTH-TRACE model)') 
st.markdown("Enter the following items to display the predicted HCC risk")

with st.form('user_inputs'): 
    age = st.number_input('Age (years)', min_value=18, max_value=100, value=60) 
    height = st.number_input('Height (cm)', min_value=100.0, max_value=300.0, value=170.0) 
    weight = st.number_input('Body weight (kg)', min_value=20.0, max_value=300.0, value=70.0)     
    PLT = st.number_input('Platelet count (×10^4/µL)', min_value=1.0, max_value=75.0, value=15.0)
    ALB = st.number_input('Albumin (g/dL)', min_value=1.0, max_value=7.0, value=4.0) 
    AST = st.number_input('AST (IU/L)', min_value=1, max_value=500, value=30)
    ALT = st.number_input('ALT (IU/L)', min_value=1, max_value=500, value=30)
    GGT = st.number_input('γ-GTP (IU/L)', min_value=1, max_value=1000, value=50)
    submitted = st.form_submit_button('Predict') 

if submitted:
    BMI = (weight / (height/100)**2)
    
    X = pd.DataFrame({
        'age': [age],
        'BMI': [BMI],
        'ALB': [ALB],
        'AST': [AST],
        'ALT': [ALT],
        'GGT': [GGT],
        'PLT': [PLT]
    })
    
    surv = rsf.predict_survival_function(X, return_array=True)
    
    plt.figure(figsize=(10, 6))
    for i, s in enumerate(surv):
        plt.step(rsf.unique_times_, s, where="post")
    
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.ylabel("Probability of HCC-free survival")
    plt.xlabel("Years")
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               ['100%', '80%', '60%', '40%', '20%', '0%'])
    
    temp_img_path = os.path.join(tempfile.gettempdir(), "hcc_prediction.png")
    plt.savefig(temp_img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    st.header("HCC Risk Prediction")
    st.image(temp_img_path)
    
    y_event = rsf.predict_survival_function(X, return_array=True).flatten()
    HCC_incidence = 100 * (1 - y_event)
    
    df_results = pd.DataFrame({
        'Time (years)': rsf.unique_times_,
        'HCC incidence (%)': HCC_incidence
    })
    
    try:
        one_year = round(df_results.iloc[174, 1], 3)
        three_year = round(df_results.iloc[459, 1], 3)
        five_year = round(df_results.iloc[740, 1], 3)
        
        st.subheader("Predicted HCC incidence")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("1-year", f"{one_year}%")
        with col2:
            st.metric("3-year", f"{three_year}%")
        with col3:
            st.metric("5-year", f"{five_year}%")
            
    except IndexError:
        st.warning("Unable to calculate specific time points")
    
    with st.expander("Detailed Results"):
        st.dataframe(df_results.head(20))
