#date: 2023-07-07T16:50:49Z
#url: https://api.github.com/gists/3ef6a2beaed48f9c2f9d861cfae39644
#owner: https://api.github.com/users/madelyn-tonic

import openai
import streamlit as st
import asyncio
import os

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent

from tonic_api.api import TonicApi

# Set up OpenAI API credentials
openai.api_key = os.environ.get('OPENAI_API_KEY')
tonic_api_key = os.environ.get('TONIC_API_KEY')

# Pull in data
tonic = TonicApi("https://app.tonic.ai", tonic_api_key)
workspace = tonic.get_workspace("WORKSPACE_ID")
model = workspace.get_trained_model_by_training_job_id("JOB_ID")
df = model.sample(30000)

# Set up agent 
llm = llm = ChatOpenAI(model_name="gpt-3.5-turbo")
agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True, handle_parsing_errors="Check your output and make sure it conforms!")


# Function to answer a simple question 
async def answer_q(question):

    prompt = question
    output = agent.run(prompt)
    return output

# Function to generate summary table
sum_prompt = PromptTemplate(
    input_variables= ["table_type", "feature_type"],
    template= "You are a data analyst. Using python_repl_ast, please provide {table_type} for the {feature_type} features in the dataframe as a table.",
)

async def summary_tbl(table, feature_typ):

    return agent.run(sum_prompt.format(table_type=table, feature_type=feature_typ))


# Create the app using Streamlit
def main():
    st.title("Query the UCI Adult Census Dataset with Tonic Data Science Mode")
    st.markdown("*This web app is powered by the `create_pandas_dataframe_agent` in Langchain. \n If you recieve an error running a query simply rerun it and the error should resolve itself.*")
    
    # Simple answer generation
    st.header("Ask a simple question of the dataset")
    question = st.text_input("Question:")
    
    if st.button("Answer Question"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(answer_q(question))
        st.text_area("Answer", value=response, height=200)

    # Generate summary tables
    st.header("Generate summary tables for a particular feature type")
    table = st.text_input("Table type:")
    feature_typ = st.text_input("Feature type:")

    if st.button("Generate tables"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(summary_tbl(table, feature_typ))
        st.text_area("Tables", value=response, height=200)

if __name__ == "__main__":
    main()