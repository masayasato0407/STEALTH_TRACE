import streamlit as st
# sksurvはscikit-survivalライブラリに含まれます
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import joblib
# ▼▼▼【変更点1】huggingface_hubをインポート ▼▼▼
from huggingface_hub import hf_hub_download

# @st.cache_resourceデコレータはそのまま使います。
# これにより、モデルのダウンロードと読み込みは初回起動時のみ実行されます。
@st.cache_resource
def load_model():
    # ▼▼▼【変更点2】ローカルからではなく、Hugging Face Hubからダウンロードして読み込む ▼▼▼
    # joblib.load(open("rsfmodel.sav", 'rb')) の行を以下に置き換えます。
    
    # hf_hub_download を使ってファイルをダウンロード
    # repo_idには、ステップ1で確認した「あなたのユーザー名/リポジトリ名」を正確に入力してください。
    # filenameには、アップロードしたモデルファイル名を入力します。
    model_path = hf_hub_download(
        repo_id="YOUR_USERNAME/YOUR_REPO_NAME",  # 例: "TaroYamada/stealth-trace-model"
        filename="rsfmodel.sav"
    )
    
    # ダウンロードしたファイルのパスを使ってモデルを読み込む
    return joblib.load(model_path)

# --- ここから下のコードは一切変更する必要はありません ---

# モデルの読み込み（初回起動時はHugging Face Hubからダウンロードするため時間がかかります）
with st.spinner("モデルをダウンロードして読み込んでいます... (初回起動時は数分かかることがあります)"):
    rsf = load_model()

st.title('Prediction model for SLD-HCC (STEALTH-TRACE model)')
st.markdown("Enter the following items to display the predicted HCC risk")

with st.form('user_inputs'):
  age=st.number_input('age (year)', min_value=18,max_value=100)
  height=st.number_input('height (cm)', min_value=100.0,max_value=300.0, value=170.0)
  weight=st.number_input('body weight (kg)', min_value=20.0,max_value=300.0, value=65.0)
  PLT=st.number_input('Platelet count (×10^4/µL)', min_value=1.0,max_value=75.0, value=15.0)
  ALB=st.number_input('Albumin (g/dL)', min_value=1.0,max_value=7.0, value=4.0)
  AST=st.number_input('AST (IU/L)', min_value=1,max_value=500, value=30)
  ALT=st.number_input('ALT (IU/L)', min_value=1,max_value=500, value=30)
  GGT=st.number_input('γ-GTP (IU/L)', min_value=1,max_value=1000, value=50)
  st.form_submit_button()

# ユーザーが何も入力しないとheightが0になりエラーになるのを防ぐ
if height > 0:
    height2=height*height
    BMI0=weight/height2
    BMI=BMI0*10000

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

    surv = rsf.predict_survival_function(X, return_array=True)

    # グラフの描画
    fig, ax = plt.subplots()
    for i, s in enumerate(surv):
        ax.step(rsf.unique_times_, s, where="post", label=str(i))
    
    ax.set_xlim(0,10)
    ax.set_ylim(0,1)
    ax.set_ylabel("predicted HCC development")
    ax.set_xlabel("years")
    ax.grid(True)
    ax.invert_yaxis()
    ax.set_yticks([0.0, 0.2, 0.4,0.6,0.8,1.0],
                ['100%', '80%', '60%', '40%', '20%', '0%'])

    st.header("HCC risk for submitted patient")
    st.pyplot(fig) # st.imageよりst.pyplotの方が推奨されます

    # 指標の計算
    y_event = rsf.predict_survival_function(X, return_array=True).flatten()
    HCCincidence=100*(1-y_event)

    df1 = pd.DataFrame(rsf.unique_times_)
    df1.columns = ['timepoint (year)']
    df2 = pd.DataFrame(HCCincidence)
    df2.columns = ['predicted HCC incidence (%)']
    df_merge = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

    # 近似値を探す方がロバストです
    one_year_idx = (np.abs(df_merge['timepoint (year)'] - 1.0)).argmin()
    three_year_idx = (np.abs(df_merge['timepoint (year)'] - 3.0)).argmin()
    five_year_idx = (np.abs(df_merge['timepoint (year)'] - 5.0)).argmin()
    
    one = round(df_merge.iloc[one_year_idx, 1], 3)
    three = round(df_merge.iloc[three_year_idx, 1], 3)
    five = round(df_merge.iloc[five_year_idx, 1], 3)
    
    st.subheader("predicted HCC incidence at each time point")
    st.write(f"**predicted HCC incidence at 1 year:** {one}%")
    st.write(f"**predicted HCC incidence at 3 year:** {three}%")
    st.write(f"**predicted HCC incidence at 5 year:** {five}%")
