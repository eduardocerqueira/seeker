#date: 2023-12-22T16:48:39Z
#url: https://api.github.com/gists/a7e9bbeec172565169c26770bb1d86b0
#owner: https://api.github.com/users/sathishgang-db

# Databricks notebook source
# MAGIC %pip install jsonformer
# MAGIC %pip install --upgrade git+https://github.com/huggingface/transformers
# MAGIC %pip install auto-gptq
# MAGIC %pip install autoawq
# MAGIC %pip install optimum
# MAGIC %pip install accelerate>=0.22.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC If cuda update is needed - Use this [link](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)

# COMMAND ----------

import os
cuda_path = '/usr/local/cuda-11.8/lib64'
# Check if LD_LIBRARY_PATH is already set
if 'LD_LIBRARY_PATH' in os.environ:
    # Append the new path to the existing LD_LIBRARY_PATH
    os.environ['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH'] + ":" + cuda_path
else:
    # Set LD_LIBRARY_PATH to the new path
    os.environ['LD_LIBRARY_PATH'] = cuda_path

# Optionally, print the updated LD_LIBRARY_PATH to verify
print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])

# COMMAND ----------

reviewDF = spark.sql("SELECT * FROM main.sgfs.costumes_amazon_cleaned")
display(reviewDF)

# COMMAND ----------

from types import SimpleNamespace
first_row = reviewDF.first()
sample_row = first_row.asDict()
sample_row = SimpleNamespace(**sample_row)

# COMMAND ----------

print(f"Sample Review is as shown below for the product : {sample_row.product_name} **** \n {sample_row.text.strip()}")

# COMMAND ----------

from jsonformer import Jsonformer
# https://github.com/1rgs/jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer,GPTQConfig
import warnings

# Set the warning action to ignore
warnings.filterwarnings('ignore')
# https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GPTQ
model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
# gptq_config = GPTQConfig(bits=4,use_exllama=False)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main",
                                             )

tokenizer = "**********"=True)
system_message = "You are a helpful assistant who parses reviews for products into JSON."
prompt = f"""Extract in JSON the sentiment of the user review (postive, negative, neutral), 
does the review mention fit issues (yes/no),
does the review mention pricing issues (yes/no),
did the product meet customer expectations (yes/no) and a 
10 word summary of the review
based on the user review below.
Review:
{sample_row.text.strip()}
"""
prompt_template=f'''<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
'''
json_schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "fit_issues": {"type": "boolean"},
        "pricing_issues": {"type": "boolean"},
        "met_customer_expectations": {"type": "boolean"},
        "summary":{"type": "string"}
        }
    }
jsonformer = "**********"
generated_data = jsonformer()
print(generated_data)

# COMMAND ----------


from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id=model_name_or_path)

# COMMAND ----------

print(prompt)

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

class Mistral7BReviewParser(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """Method to initialize the model and tokenizer."""
        # model_name_or_path = "TheBloke/Mistral-7B-OpenOrca-GPTQ"
        self.model = AutoModelForCausalLM.from_pretrained(
          context.artifacts['repository'],
            device_map="auto",
            trust_remote_code=False
        )
        self.tokenizer = "**********"=True)
        self.model.eval()
        self.system_message = "You are a helpful assistant who parses reviews for products into JSON."
        self.json_schema = {
        "type": "object",
        "properties": {
        "sentiment": {"type": "string"},
        "fit_issues": {"type": "boolean"},
        "pricing_issues": {"type": "boolean"},
        "met_customer_expectations": {"type": "boolean"},
        "summary":{"type": "string"}
                    }
        }
        # self.jsonformer = "**********"
    def _build_prompt(self, instruction):
        """this method is used to build prompt for a single input"""
        return f'''<|im_start|>system
        {self.system_message}<|im_end|>
        <|im_start|>user
        {instruction}<|im_end|>
        <|im_start|>assistant
        '''
    
    def _generate_response(self, prompt):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        prompt = self._build_prompt(prompt)
        # Call JSONFormer
        jsonformer = "**********"
        generated_response = jsonformer()
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
    "prompt":[prompt],
})

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=Mistral7BReviewParser(),
        artifacts={'repository' : snapshot_location},
        input_example=input_example,
        pip_requirements=["torch==2.0.1","transformers==4.35.2", "cloudpickle==2.0.0","jsonformer==0.12.0","accelerate>=0.22.0","torchvision==0.15.2","auto-gptq==0.5.1","optimum"],
        signature=signature
    )

# COMMAND ----------

import mlflow
#register model to UC
mlflow.set_registry_uri("databricks-uc")
registered_name = "main.sgfs.mistral_7b_reviewparse" 
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    registered_name,
)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
# Annotate the model as "CHAMPION".
client.set_registered_model_alias(name=registered_name, alias="Champion", version=result.version)
# Load it back from UC
loaded_model = mlflow.pyfunc.load_model(f"models:/{registered_name}@Champion")

# COMMAND ----------

import json
json.dumps(loaded_model.predict({'prompt': prompt})).encode()

# COMMAND ----------

[json.dumps(candidate) for candidate in loaded_model.predict({'prompt': prompt})['candidates']]

# COMMAND ----------

# MAGIC %md let's now score the entire reviews dataset

# COMMAND ----------

from pyspark.sql.functions import concat_ws, lit

# Define the string to be concatenated with the 'text' column
instruction = f"""Extract in JSON the sentiment of the user review (positive, negative, neutral), 
does review mention fit issues (yes / no / not-in-review),
does the review mention pricing issues (yes / no / not-in-review),
did the product meet customer expectations (yes / no / not-enough-info) and a 
10 word summary of the review
based on the user review below.
Review:
"""

# Add the new 'instructions' column by concatenating the 'instruction' string
# with the 'text' column
reviewDF = (reviewDF
             .withColumn('instructions', 
                         concat_ws('\n', lit(instruction), reviewDF.text))
            )

# Show the resulting dataframe
display(reviewDF)

# COMMAND ----------


from typing import Iterator
import pandas as pd
from pyspark.sql.functions import pandas_udf, PandasUDFType, col
from typing import Dict

# COMMAND ----------

def score_review(df):
  import json
  # df[result] = loaded_model.predict({'prompt': df['instructions']})
  df['result'] = [json.dumps(candidate) for candidate in loaded_model.predict({'prompt': df['instructions']})['candidates']]
  return df

# COMMAND ----------

from pyspark.sql.functions import lit
testDF = reviewDF.limit(20)
testDF = testDF.withColumn('ID', lit(1))
display(testDF)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType,MapType,ArrayType
output_schema = StructType([
    StructField('text', StringType(), True),
    StructField('date', StringType(), True),
    StructField('title', StringType(), True),
    StructField('rating', LongType(), True),
    StructField('product_name', StringType(), True),
    StructField('instructions', StringType(), False),
    StructField('ID', IntegerType(), False),
    StructField('result', StringType(), True)
])

# COMMAND ----------

result_df = testDF.groupBy('ID').applyInPandas(score_review,schema=output_schema)

# COMMAND ----------

display(result_df)

# COMMAND ----------

# MAGIC %md Deploy as an endpoint

# COMMAND ----------

endpoint_name = 'review-parser-sg'

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

# COMMAND ----------

result_df.write.mode('overwrite').saveAsTable('main.sgfs.sample_results')

# COMMAND ----------

reviewDF = reviewDF.withColumn('ID', lit(1)).limit(500)
full_df = reviewDF.groupBy('ID').applyInPandas(score_review,schema=output_schema)

# COMMAND ----------

full_df.write.mode('overwrite').saveAsTable('main.sgfs.review_results')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from main.sgfs.review_results;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from main.sgfs.review_results;

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table main.sgfs.review_analysis as
# MAGIC (
# MAGIC select product_name, 
# MAGIC text as customer_review,
# MAGIC rating as product_rating,
# MAGIC date as review_date, 
# MAGIC result:sentiment as sentiment,
# MAGIC case when result:fit_issues='true' then 1 else 0 end fit_issues, 
# MAGIC case when result:pricing_issues='true' then 1 else 0 end  pricing_issues,
# MAGIC case when result:met_customer_expectations='true' then 1 else 0 end met_customer_expectations,
# MAGIC result:summary as summary 
# MAGIC from main.sgfs.review_results
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from main.sgfs.review_analysis;

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we have to come up with a way to ask questions off this da
# MAGIC - most liked product
# MAGIC - least liked product
# MAGIC - product with most fit issues
# MAGIC - product with most pricing issues
# MAGIC - product that didn't meet customer expectations
# MAGIC - some good exploration of the product summary
# MAGIC

# COMMAND ----------ow, we have to come up with a way to ask questions off this da
# MAGIC - most liked product
# MAGIC - least liked product
# MAGIC - product with most fit issues
# MAGIC - product with most pricing issues
# MAGIC - product that didn't meet customer expectations
# MAGIC - some good exploration of the product summary
# MAGIC

# COMMAND ----------