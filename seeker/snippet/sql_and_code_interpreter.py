#date: 2025-02-25T17:10:48Z
#url: https://api.github.com/gists/3b17ff5bea1b02c0bc3cfa234a83adc8
#owner: https://api.github.com/users/vhoudebine

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool
from azure.ai.projects.models import FilePurpose, MessageRole, MessageAttachment
from azure.ai.projects.models import FunctionTool, RequiredFunctionToolCall, SubmitToolOutputsAction, ToolOutput
from azure.identity import DefaultAzureCredential
from pathlib import Path
from dotenv import load_dotenv
import pyodbc, struct
import pandas as pd
import json
import warnings
import time
from colorama import Fore, Style, init
import os

load_dotenv()
warnings.filterwarnings('ignore')


server = os.getenv('AZURE_SQL_SERVER') 
database = os.getenv('AZURE_SQL_DB_NAME')
username = os.getenv('AZURE_SQL_USER') 
password = "**********"

#connection_string = f'Driver={{ODBC Driver 18 for SQL Server}};Server=tcp: "**********"
connection_string = f'Driver={{ODBC Driver 18 for SQL Server}};Server=tcp:{server},1433;Database={database};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'

def get_conn():
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=False)
    token_bytes = credential.get_token("https: "**********"
    token_struct = "**********"
    SQL_COPT_SS_ACCESS_TOKEN = "**********"
    conn = pyodbc.connect(connection_string, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: "**********"
    return conn

engine_azure = get_conn()

def convert_datetime_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any datetime and timestamp columns in a DataFrame to strings."""
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].astype(str)
    return df

def query_azure_sql(query: str) -> str:
    """Run a SQL query on Azure SQL and return results as a pandas DataFrame
    :param query: str the SQL query to execute
    :return: the path of the file containing the results
    :rtype: str
    """
    print(f"Executing query on Azure SQL: {query}")
    df = pd.read_sql(query, engine_azure)
    df = convert_datetime_columns_to_string(df)
    return json.dumps(df.to_dict(orient='records'))

# Create an Azure AI Client from a connection string, copied from your Azure AI Foundry project.
# At the moment, it should be in the format "<HostName>;<AzureSubscriptionId>;<ResourceGroup>;<HubName>"
# Customer needs to login to Azure subscription via Azure CLI and set the environment variables
project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(), conn_str=os.environ["project_connection_string"]
)

# Initialize function tool with user functions
functions = FunctionTool(functions=[query_azure_sql])




code_interpreter = CodeInterpreterTool()

# Create agent with code interpreter tool and tools_resources
agent = project_client.agents.create_agent(
    model="gpt-4o-global",
    name="my-assistant-2",
    instructions="You are helpful assistant",
    tools=code_interpreter.definitions+functions.definitions,
    #tools = functions.definitions
    tool_resources=code_interpreter.resources,
)
# [END upload_file_and_create_agent_with_code_interpreter]
print(f"Created agent, agent ID: {agent.id}")

thread = project_client.agents.create_thread()
print(f"Created thread, thread ID: {thread.id}")

# Create a message
message = project_client.agents.create_message(
    thread_id=thread.id,
    role="user",
    content="""Retrieve ProductID and Color for all records from the table SalesLT.Product,
    do not try to read the output file, just mention the path where the file is stored""",
)

run = project_client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)

if run.status == "failed":
    # Check if you got "Rate limit is exceeded.", then you want to get more quota
    print(f"Run failed: {run.last_error}")

while run.status in ["queued", "in_progress", "requires_action"]:
    time.sleep(1)
    run = project_client.agents.get_run(thread_id=thread.id, run_id=run.id)

    # Execute any tool call. In our case if the tool call is query_azure_sql, 
    # we will save the output as a CSV file and upload it to the agent thread
    if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
        messages = project_client.agents.list_messages(thread_id=thread.id)
        print(Fore.GREEN + f"Assistant: {messages.get('data')[0].get('content')[0]['text']['value']}" + Style.RESET_ALL)
        tool_calls = run.required_action.submit_tool_outputs.tool_calls
        if not tool_calls:
            print("No tool calls provided - cancelling run")
            project_client.agents.cancel_run(thread_id=thread.id, run_id=run.id)
            break

        tool_outputs = []
        for tool_call in tool_calls:
            if isinstance(tool_call, RequiredFunctionToolCall):
                try:
                    print(f"Executing tool call: {tool_call}")
                    output = functions.execute(tool_call)
                    if tool_call.function.name == "query_azure_sql":
                        # Parse the output and save it to a CSV file
                        output_parsed = json.loads(output)
                        df = pd.DataFrame(output_parsed)
                        output_dir = Path("./tmp")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        file_path = f"./tmp/query_output_{tool_call.id}.csv"
                        df.to_csv(file_path, index=False)

                        print("uploading file to Azure AI Foundry project")
                        file = project_client.agents.upload_file_and_poll(
                        file_path=file_path, purpose=FilePurpose.AGENTS
                )
                        
                    tool_outputs.append(
                        ToolOutput(
                            tool_call_id=tool_call.id,
                            output=f"saved to file ./tmp/query_output_{tool_call.id}.csv",
                        )
                    )
                except Exception as e:
                    print(f"Error executing tool_call {tool_call.id}: {e}")

        print(f"Tool outputs: {tool_outputs}")
        if tool_outputs:
            project_client.agents.submit_tool_outputs_to_run(
                thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
            )
            attachment = MessageAttachment(file_id=file.id, tools=code_interpreter.definitions)

    print(f"Current run status: {run.status}")

# We have to let the first run complete before we can start the second one 
# as you can't add a file to a thread while a run is in progress
print(f"First run completed with status: {run.status}")

# Create a message to run the code interpreter session on the file 
message = project_client.agents.create_message(
                thread_id=thread.id, 
                role="user", 
                content="Use code interpreter to Plot the color distribution from the provided file", attachments=[attachment]
        )

run = project_client.agents.create_run(thread_id=thread.id, assistant_id=agent.id)

# Poll the run as long as run status is queued or in progress
while run.status in ["queued", "in_progress", "requires_action"]:
    # Wait for a second
    time.sleep(1)
    run = project_client.agents.get_run(thread_id=thread.id, run_id=run.id)
    messages = project_client.agents.list_messages(thread_id=thread.id)

# When second run is done, retrieve the messages and print the results
messages = project_client.agents.list_messages(thread_id=thread.id)

for mess in messages.get('data')[0].get('content')[0]:
    if 'text' in mess:
        print(Fore.GREEN + f"Assistant: {mess['text']['value']}" + Style.RESET_ALL)


for image_content in messages.image_contents:
    file_id = image_content.image_file.file_id
    print(f"Image File ID: {file_id}")
    file_name = f"{file_id}_image_file.png"
    project_client.agents.save_file(file_id=file_id, file_name=file_name)
    print(f"Saved image file to: {Path.cwd() / file_name}") in messages.image_contents:
    file_id = image_content.image_file.file_id
    print(f"Image File ID: {file_id}")
    file_name = f"{file_id}_image_file.png"
    project_client.agents.save_file(file_id=file_id, file_name=file_name)
    print(f"Saved image file to: {Path.cwd() / file_name}")