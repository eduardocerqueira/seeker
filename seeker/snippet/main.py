#date: 2024-03-27T17:04:58Z
#url: https://api.github.com/gists/03d1c3734e3b1a5d1032425e30842466
#owner: https://api.github.com/users/jorgemaciel

import streamlit as st
import pandas as pd
import os
from pandasai import SmartDataframe


st.title('Análise por prompt com pandas AI')

upload_file = st.file_uploader('Upload arquivo CSV para análise', type=['csv'])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.write(df.head())

    # convert to SmartDataframe
    sdf = SmartDataframe(df)

    prompt = st.text_area('Prompt')

    if st.button('Enter'):
        if prompt:
            with st.spinner('PandasAI está tentando responder, espere um pouquinho...'):
                response = sdf.chat(prompt)
                st.success(response)

                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
        else:
            st.warning('Enter prompt')