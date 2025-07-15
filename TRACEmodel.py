import streamlit as st
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import joblib
import gdown
import os
import tempfile

def download_model():
    """Download model from Google Drive if not exists"""
    temp_dir = tempfile.gettempdir()
    model_path = os.path.join(temp_dir, "rsfmodel.sav")
    
    if not os.path.exists(model_path):
        file_id = "1BUwxs50NOH3yMjhmgCvarcc4vMkjSHFa"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Downloading model...")
            gdown.download(url, model_path, quiet=False)
            progress_bar.progress(100)
            status_text.text("Model downloaded successfully!")
            return model_path
        except Exception as e:
            st.error(f"Model download failed: {e}")
            return None
    else:
        st.success("Model found in cache!")
        return model_path

@st.cache_resource 
def load_model(model_path):
    """Load model from file"""
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

st.title('Prediction model for SLD-HCC (STEALTH-TRACE model)') 
st.markdown("Enter the following items to display the predicted HCC risk")

with st.form('user_inputs'): 
    age = st.number_input('Age (years)', min_value=18, max_value=100) 
    height = st.number_input('Height (cm)', min_value=100.0, max_value=300.0) 
    weight = st.number_input('Body weight (kg)', min_value=20.0, max_value=300.0)     
    PLT = st.number_input('Platelet count (×10^4/µL)', min_value=1.0, max_value=75.0)
    ALB = st.number_input('Albumin (g/dL)', min_value=1.0, max_value=7.0) 
    AST = st.number_input('AST (IU/L)', min_value=1, max_value=500)
    ALT = st.number_input('ALT (IU/L)', min_value=1, max_value=500)
    GGT = st.number_input('γ-GTP (IU/L)', min_value=1, max_value=1000)
    submitted = st.form_submit_button('Predict') 

if submitted:
    # Only download and load model when prediction is requested
    model_path = download_model()
    
    if model_path:
        rsf = load_model(model_path)
        
        if rsf:
            # Calculate BMI using original formula
            height2 = height * height
            BMI0 = weight / height2
            BMI = BMI0 * 10000
            
            X = pd.DataFrame(
                data={'age': [age],
                      'BMI': [BMI],
                      'ALB': [ALB],
                      'AST': [AST],
                      'ALT': [ALT],
                      'GGT': [GGT],
                      'PLT': [PLT],
                     }
            )
            
            try:
                # Generate prediction
                surv = rsf.predict_survival_function(X, return_array=True)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                for i, s in enumerate(surv):
                    ax.step(rsf.unique_times_, s, where="post")
                
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability of HCC-free survival")
                ax.set_xlabel("Years")
                ax.grid(True)
                ax.invert_yaxis()
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['100%', '80%', '60%', '40%', '20%', '0%'])
                
                temp_img_path = os.path.join(tempfile.gettempdir(), "hcc_prediction.png")
                fig.savefig(temp_img_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                st.header("HCC Risk Prediction")
                st.image(temp_img_path)
                
                # Calculate HCC incidence
                y_event = rsf.predict_survival_function(X, return_array=True).flatten()
                HCCincidence = 100 * (1 - y_event)
                
                df1 = pd.DataFrame(rsf.unique_times_)
                df1.columns = ['timepoint (year)']
                df2 = pd.DataFrame(HCCincidence)
                df2.columns = ['predicted HCC incidence (%)']
                df_merge = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
                
                # Display key time points
                try:
                    if len(df_merge) > 740:
                        one0 = df_merge.iloc[174, 1]
                        one = round(one0, 3)
                        three0 = df_merge.iloc[459, 1]
                        three = round(three0, 3)
                        five0 = df_merge.iloc[740, 1]
                        five = round(five0, 3)
                        
                        st.subheader("Predicted HCC incidence")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("1-year", f"{one}%")
                        with col2:
                            st.metric("3-year", f"{three}%")
                        with col3:
                            st.metric("5-year", f"{five}%")
                    else:
                        st.warning("Insufficient data points for specific time predictions")
                        
                except IndexError as e:
                    st.warning(f"Unable to calculate specific time points: {e}")
                
                # Show detailed results
                with st.expander("Detailed Results"):
                    st.dataframe(df_merge.head(20))
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Please check your input values and try again.")
        else:
            st.error("Failed to load model. Please try again.")
    else:
        st.error("Failed to download model. Please try again.")
