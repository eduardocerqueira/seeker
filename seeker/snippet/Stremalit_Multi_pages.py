#date: 2021-12-20T17:14:06Z
#url: https://api.github.com/gists/81a4884e3f3aea7e11462ca0bf289927
#owner: https://api.github.com/users/dormeir999

# Helper functions
def data_preprocess(state, process_table_sidebar):
    autosave_session(state)
    st.title("Data Preprocess")
    ...
def training(state, process_table_sidebar):
    autosave_session(state)
    st.title("Train Model")
    ...
... # evaluation, prediction, save_and_load, free_coding, deployment
# Implementation
def main(state)
    pages = {
        "Preprocessing": data_preprocess,
        "Training": training,
        "Evaluation": evaluation,
        "Prediction": prediction,
        "Save & Load": save_and_load,
        "Free Coding": free_coding,    
        "Deployment": deployment}
state.page = st.sidebar.radio("CRISP-DM", 
    tuple(pages.keys()),                             
    index=tuple(pages.keys()).index(state.page) if state.page
                                                else   0)
   if st.sidebar.button("Logout"):
      state.clear()