#date: 2025-11-26T17:01:31Z
#url: https://api.github.com/gists/717e7307bc2e96f6d90c9918c89c35be
#owner: https://api.github.com/users/themysterysolver

import streamlit as st
from logic import algoSolver
import pandas as pd

st.title('CPU scheduling algorithm')


at = st.text_input('Arrival time')
bt = st.text_input('Burst time')


option = st.selectbox(
    "Algorithm",
    ("First come first served(FCFS)", "Shortest job first(SJF)", "Shortest remaining time first(SRTF)","Premptive Priority scheduling",
     "Non-Premptive priority scheduling","Round robin(RR)"),
    index=None,
    placeholder="Choose an algorithm"
)

tq = None
priority = None

if option == "Round robin(RR)":
    tq = st.text_input('Time quantum')
elif option == "Premptive Priority scheduling" or option == "Non-Premptive priority scheduling":
    priority = st.text_input('Priority',placeholder="Lower =# Higher priority")


if st.button('Solve'):
    ans = algoSolver(at,bt,option,tq,priority)
    if not ans:
        st.error("Incorrect input")
    else:
        headers = ans.keys()
        data = dict()
        for key in headers:
            if key!="Process":
                data[key] = ans[key]
        df = pd.DataFrame(data,index=ans['Process'])
        st.table(df)



