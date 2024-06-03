#date: 2024-06-03T16:43:03Z
#url: https://api.github.com/gists/3ff5e0c7bcf0da0b33ce96576027e703
#owner: https://api.github.com/users/Dardrich

import streamlit as st

st.write('# Streamlit calculator')
number1= st.number_input('number 1')
number2 = st.number_input('number 2')
num3 = number1+number2
st.write('# Answer is ',num3)