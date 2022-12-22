#date: 2022-12-22T16:57:00Z
#url: https://api.github.com/gists/626d45735767164efc2d43819e1d63cb
#owner: https://api.github.com/users/StatsGary

 # Create text prompt
 prompt = st.text_input('Input the prompt desired')
 # Checks if the example prompts has been chosen and overides the text input
 if add_selectbox != 'None' or prompt is None:
    prompt = add_selectbox