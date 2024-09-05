#date: 2024-09-05T16:57:47Z
#url: https://api.github.com/gists/8818dafe04307567ce0df7c318a9bd04
#owner: https://api.github.com/users/louspringer

# This code snippet creates a Snowflake Snowpark session. It works equivalently to obtain a 
# session in a Snowflake hosted notebook, python script or Streamlit application, without
# the ~/.snowsql/config configuration.
#
# It uses the 'connection_name' set to 'default' to configure and get or create the session.
# The session is initialized in a concise one-liner.
# 
# Local Configuration:
# Ensure the following is set up in the ~/.snowsql/config file:
#
# [connections.default]
# accountname = <your_account>
# username = <your_username>
# private_key_path = <path_to_private_key>  # Optional if you're using keypair auth
# warehouse = <your_warehouse>
# database = <your_database>
# schema = <your_schema>
# role = <your_role>
#
# Replace <placeholders> with your actual Snowflake account details.
# The 'default' profile will pull these connection parameters automatically when creating the session.

from snowflake.snowpark import Session
session = Session.builder.config(key="connection_name", value="default").getOrCreate()
