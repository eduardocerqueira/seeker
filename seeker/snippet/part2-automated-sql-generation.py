#date: 2023-12-22T16:48:39Z
#url: https://api.github.com/gists/a7e9bbeec172565169c26770bb1d86b0
#owner: https://api.github.com/users/sathishgang-db

# Databricks notebook source
# import os
# cuda_path = '/usr/local/cuda-12.3/lib64'
# # Check if LD_LIBRARY_PATH is already set
# if 'LD_LIBRARY_PATH' in os.environ:
#     # Append the new path to the existing LD_LIBRARY_PATH
#     os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":" + cuda_path
# else:
#     # Set LD_LIBRARY_PATH to the new path
#     os.environ['LD_LIBRARY_PATH'] = cuda_path

# # Optionally, print the updated LD_LIBRARY_PATH to verify
# print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])

# COMMAND ----------

# MAGIC %pip install jsonformer
# MAGIC %pip install --upgrade transformers
# MAGIC %pip install --upgrade auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
# MAGIC %pip install autoawq
# MAGIC %pip install optimum
# MAGIC %pip install accelerate>=0.22.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip show transformers

# COMMAND ----------

# MAGIC %pip show torch

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/sqlcoder-34b-alpha-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-128g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="cuda:0",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = "**********"=True)

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog main

# COMMAND ----------

# MAGIC %sql
# MAGIC use schema sgfs

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from review_analysis;

# COMMAND ----------

database = "main"
table = "sgfs.review_analysis"
column_schema = spark.sql(f"describe table {database}.{table}").collect()

# construct CREATE TABLE DDL
ddl = f"CREATE TABLE IF NOT EXISTS {database}.{table} (\n"
for i, column in enumerate(column_schema):
    # ignore the last column since it contains all NULL values
    if i == len(column_schema):
        break
    name, dtype, comment = column  
    ddl += f"  {name} {dtype.upper()}{',' if i != len(column_schema)-1 else ''} --{comment}\n"
ddl = ddl[:-2] + "\n)" # remove the last comma and newline

print(ddl)

# COMMAND ----------

prompt = "Top Products that have fit issues?"
prompt_template=f'''### Task
Generate a SQL query to answer the following question:
`{prompt}`

### Database Schema
This query will run on a database whose schema is represented in this string:
{ddl};

### SQL
Given the database schema, here is the SQL query that answers `{prompt}`:
```sql
'''
input_ids = "**********"="pt").input_ids.cuda()
eos_token_id = "**********"
output = model.generate(
    inputs=input_ids,
    num_return_sequences=1,
    eos_token_id= "**********"
    pad_token_id= "**********"
    do_sample=False,
    num_beams=1,
    max_new_tokens= "**********"
)
result = "**********"

# COMMAND ----------

import re
sql_query_match = re.search(r"```sql\s*(.*?)\s*```", result, re.DOTALL)
if sql_query_match:
    extracted_sql = sql_query_match.group(1)
else:
  print("no sql found")

print(extracted_sql)

# COMMAND ----------

# What is the average product rating by products?

# COMMAND ----------

from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id=model_name_or_path)

# COMMAND ----------

prompt_template

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

class SQLCoder34BGPTQ(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """Method to initialize the model and tokenizer."""

        self.model = AutoModelForCausalLM.from_pretrained(
          context.artifacts['repository'],
            device_map="auto",
            trust_remote_code=False
        )
        self.tokenizer = "**********"=True)
        self.model.eval()
    
    def _generate_response(self, prompt):
        """
        This method generates prediction for a single input.
        """
        import re
        # Build the prompt
        # Send to model and generate a response
        input_ids = "**********"="pt").input_ids.cuda()
        eos_token_id = "**********"
        output = model.generate(
            inputs=input_ids,
            num_return_sequences=1,
            eos_token_id= "**********"
            pad_token_id= "**********"
            do_sample=False,
            num_beams=1,
            max_new_tokens= "**********"
        )
        result = "**********"
        #use the result and extract SQL
        sql_query_match = re.search(r"```sql\s*(.*?)\s*```", result, re.DOTALL)
        if sql_query_match:
            generated_response = sql_query_match.group(1)
        return generated_response

    def predict(self, context, model_input):
        """Method to generate predictions for the given input."""
        outputs = []
        for i in range(len(model_input)):
            prompt = model_input["prompt"][i]
            generated_data = self._generate_response(prompt)
            outputs.append(generated_data)
        return {"candidates": outputs}

# Define input and output schema for the model
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

input_schema = Schema([ColSpec(DataType.string, "prompt")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame({
    "prompt":[prompt_template],
})

# COMMAND ----------

# MAGIC
# MAGIC %pip show torchvision

# COMMAND ----------

# Log the model using MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=SQLCoder34BGPTQ(),
        artifacts={'repository' : snapshot_location},
        input_example=input_example,
        pip_requirements=["torch==2.0.1","transformers==4.36.1", "cloudpickle==2.0.0","accelerate>=0.22.0","torchvision==0.15.2","auto-gptq==0.6.0","optimum"],
        signature=signature
    )

# COMMAND ----------

import mlflow
#register model to UC
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = "main.sgfs.sqlcoder34b"
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    MODEL_NAME,
)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
# # Annotate the model as "CHAMPION".
client.set_registered_model_alias(name=MODEL_NAME, alias="Champion", version=result.version)
# Load it back from UC
import mlflow
loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")

# COMMAND ----------

# Make a prediction using the loaded model
loaded_model.predict(
    {
        "prompt": [prompt_template],
    }
)

# COMMAND ----------

# Another prediction
loaded_model.predict(
    {
        "prompt": ["""### Task\n    Generate a SQL query to answer the following question:\n    `Average Product Rating?`\n\n    ### Database Schema\n    This query will run on a database 
whose schema is represented in this string:\n    CREATE TABLE IF NOT EXISTS main.sgfs.review_analysis (\n    product_name STRING, --The name of the product being reviewed.\n    customer_review 
STRING, --The text of the customer review.\n    product_rating BIGINT, --The rating given by the customer.\n    review_date STRING, --The date the review was posted.\n    sentiment STRING, --The 
sentiment analysis of the customer review.\n    fit_issues STRING, --Any fit issues mentioned in the customer review.\n    pricing_issues STRING, --Any pricing issues mentioned in the customer 
review.\n    met_customer_expectations STRING, --Whether or not the product met the customer's expectations.\n    summary STRING --A brief overview of the customer review.);\n\n    ### SQL\n    
Given the database schema, here is the SQL query that answers `Average Product Rating?`:\n    ```sql\n"""],
    }
)

# COMMAND ----------

# MAGIC %md Deploying to model serving

# COMMAND ----------

endpoint_name = 'sql-coder-34b-sg'

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = "**********"

import requests
import json

deploy_headers = {'Authorization': "**********": 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model
served_name = f'{model_version.name.replace(".", "_")}_{model_version.version}'

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU_LARGE"

endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": served_name,
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": workload_type,
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')
# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)
if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# COMMAND ----------

print(deploy_response.json())

# COMMAND ----------od='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)
if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# COMMAND ----------

print(deploy_response.json())

# COMMAND ----------