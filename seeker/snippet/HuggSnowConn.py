#date: 2023-02-07T17:06:09Z
#url: https://api.github.com/gists/a53f23b33ff0085b4eb2cb411b5e4ac6
#owner: https://api.github.com/users/daanalytics

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    #return snowflake.connector.connect(**st.secrets["snowflake"])
    return snowflake.connector.connect(
            user        = "**********"
            password    = "**********"
            account     = "**********"
            role        = "**********"
        )

# Create context 
def create_sf_session_object():

    if "snowflake_context" not in st.session_state:

        ctx = init_connection()
        
        st.session_state['snowflake_context'] = ctx

    else: 

        ctx = st.session_state['snowflake_context']

    return ctxnowflake_context']

    return ctx