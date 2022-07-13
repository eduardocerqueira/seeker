#date: 2022-07-13T17:13:42Z
#url: https://api.github.com/gists/4cc44b5f476dc80564406c0774118679
#owner: https://api.github.com/users/josephlewisjgl

# set up options
with st.sidebar:
    '''
    ### Filter options
    '''
    FILTER_OPEN = st.selectbox('Show open only: ', (True, False))
    
# filter logic 
if FILTER_OPEN:
    df = df[df['is_open'] == True]