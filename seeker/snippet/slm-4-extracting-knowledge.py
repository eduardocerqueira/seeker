#date: 2023-12-22T16:43:44Z
#url: https://api.github.com/gists/ac9a18b90c726f6a297f0c5163bd29d3
#owner: https://api.github.com/users/sathishgang-db

# Databricks notebook source
!pip install jsonformer
!pip install --upgrade git+https://github.com/huggingface/transformers
!pip install auto-gptq
!pip install optimum
!pip install python-dotenv

# COMMAND ----------

 dbutils.library.restartPython()

# COMMAND ----------

from dotenv import load_dotenv
load_dotenv(".env")
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")
import os
login(token= "**********"

# COMMAND ----------

from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name_or_path = "TheBloke/tinyllama-1.1b-chat-v0.3_platypus-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = "**********"=True)
product = "Orange 18mg 15ml High Liq"
prompt = f"""Generate a single JSON array based on the schema provided. Always provide units when extracting strength and capacity.
Product Descriptions:
{product}
"""
prompt_template=f'''### User:
{prompt}

### Response:
'''
json_schema = {
  "type" : "object",
  "properties": {
    "name": {"type": "string"},
    "flavor": {"type": "string"},
    "strength": {"type": "string"},
    "capacity": {"type": "string"},
    "is_flavored":{"type": "boolean"},
    "is_refill":{"type": "boolean"},
    "is_kit":{"type": "boolean"}
  }
}
jsonformer = "**********"
generated_data = jsonformer()
print(generated_data)Jsonformer(model, tokenizer, json_schema, prompt_template)
generated_data = jsonformer()
print(generated_data)