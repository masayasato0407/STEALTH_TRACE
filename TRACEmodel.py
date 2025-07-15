import streamlit as st
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import joblib

@st.cache_resource 
def load_model():
    return joblib.load(open("rsfmodel.sav", 'rb'))

rsf = load_model()

st.title('Prediction model for SLD-HCC (STEALTH-TRACE model)') 
st.markdown("Enter the following items to display the predicted HCC risk")

with st.form('user_inputs'): 
  age=st.number_input('age (year)', min_value=18,max_value=100) 
  height=st.number_input('height (cm)', min_value=100.0,max_value=300.0) 
  weight=st.number_input('body weight (kg)', min_value=20.0,max_value=300.0)     
  PLT=st.number_input('Platelet count (×10^4/µL)', min_value=1.0,max_value=75.0)
  ALB=st.number_input('Albumin (g/dL)', min_value=1.0,max_value=7.0) 
  AST=st.number_input('AST (IU/L)', min_value=1,max_value=500)
  ALT=st.number_input('ALT (IU/L)', min_value=1,max_value=500)
  GGT=st.number_input('γ-GTP (IU/L)', min_value=1,max_value=1000)
  st.form_submit_button() 

height2=height*height
BMI0=weight/height2
BMI=BMI0*10000

surv = rsf.predict_survival_function(pd.DataFrame(
    data={'age': [age],
          'BMI': [BMI],
          'ALB': [ALB],
          'AST': [AST],
          'ALT': [ALT],
          'GGT': [GGT],
          'PLT': [PLT],
         }
), return_array=True)

for i, s in enumerate(surv):
    plt.step(rsf.unique_times_, s, where="post", label=str(i))
plt.xlim(0,10)
plt.ylim(0,1)
plt.ylabel("predicted HCC development")
plt.xlabel("years")
plt.grid(True)

plt.gca().invert_yaxis()
plt.yticks([0.0, 0.2, 0.4,0.6,0.8,1.0],
            ['100%', '80%', '60%', '40%', '20%', '0%'])
plt.savefig("img.png")

st.header("HCC risk for submitted patient")
st.image ("img.png")

X=pd.DataFrame(
    data={'age': [age],
          'BMI': [BMI],
          'ALB': [ALB],
          'AST': [AST],
          'ALT': [ALT],
          'GGT': [GGT],
          'PLT': [PLT],
         }
)

y_event = rsf.predict_survival_function(X, return_array=True).flatten()

HCCincidence=100*(1-y_event)

df1 = pd.DataFrame(rsf.unique_times_)
df1.columns = ['timepoint (year)']
df2 = pd.DataFrame(HCCincidence)
df2.columns = ['predicted HCC incidence (%)']
df_merge = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

one0=df_merge.iloc[174,1]
one=round(one0, 3)
three0=df_merge.iloc[459,1]
three=round(three0, 3)
five0=df_merge.iloc[740,1]
five=round(five0, 3)

st.subheader("predicted HCC incidence at each time point")
st.write(f"**predicted HCC incidence at 1 year:** {one}%")
st.write(f"**predicted HCC incidence at 3 year:** {three}%")
st.write(f"**predicted HCC incidence at 5 year:** {five}%")
