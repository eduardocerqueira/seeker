#date: 2025-06-27T17:12:51Z
#url: https://api.github.com/gists/45b105cf93ab39e986f1490a7ebbee2a
#owner: https://api.github.com/users/eric-burel

# -*- coding: utf-8 -*-
"""LBKE Anthropic batch runner.ipynb

## Demo of calling Anthropic Batch API from a LangGraph graph

Improvements:
- an interrupt should be prefered to ending the chart when the work is pending => currently each new run to check if the batch is there will send a new message to the thread since we re-invoke the chart
- better processing responses, maybe using a MessageState/add reducer to treat the response as an assistant message added to a conversation
- as a bonus, polling the chart on a predefined frequence (useful for a quick run)
- test if it still works after the notebook is closed thanks to gdrive checkpointing
"""

# Commented out IPython magic to ensure Python compatibility.
# gist: https://gist.github.com/eric-burel/97156eea3f865104f0815c1aa143af19
# %pip install -q langgraph
# %pip install -q langgraph-checkpoint-sqlite
# https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
# %pip install -q anthropic
# %pip install -qU langsmith

from langsmith import traceable
import os
from google.colab import userdata
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=userdata.get("LANGSMITH_API_KEY")

from google.colab import userdata
anthropic_api_key=userdata.get('ANTHROPIC_API_KEY')
os.environ["ANTHROPIC_API_KEY"]=anthropic_api_key

import anthropic
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
client = anthropic.Anthropic()

from google.colab import drive
DRIVE_MOUNT_POINT = "/content/drive"
drive.mount(DRIVE_MOUNT_POINT)

os.listdir("/content/drive")

YOUR_DRIVE="Shareddrives/<YOUR FOLDER>" # for shared drives
# YOUR_DRIVE="MyDrive/<change this value>" # For non shared drives
AGENT_NAME="anthropic_batch_runner"

CHECKPOINT_FOLDER=os.path.join(DRIVE_MOUNT_POINT, YOUR_DRIVE, "checkpoints", AGENT_NAME)
print(CHECKPOINT_FOLDER)

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
conn = sqlite3.connect(os.path.join(CHECKPOINT_FOLDER, "checkpoints.sqlite"), check_same_thread=False)
memory = SqliteSaver(conn)

# prompt: compute timestamp for current date, as a function, outputs a string

import datetime

def get_timestamp_string():
    """
    Returns the current timestamp as a string.
    """
    return datetime.datetime.now().strftime("%m%d%H%M%S")

print(get_timestamp_string())

from langgraph.graph import StateGraph, START, END
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

# TODO: use different output/input types
class State(TypedDict):
    batch: any
    ids: list[str]
    inputs: list[str]
    results: list[any]

def decide_running(state: State):
  if state.get("batch", None) is None:
    return "batch_runner"
  else:
      batch_id=state["batch"].id
      message_batch = client.messages.batches.retrieve(
         batch_id
      )
      status=message_batch.processing_status
      print(f"Batch id {batch_id} status: {status}")
      if status == "ended":
        return "get_results"
      else:
        return END

def batch_runner(state: State, config: dict):
    if len(state["inputs"]) == 0:
      raise Exception("Inputs can't be empty")
    # Keep track of ids to be able to reorder later on
    ids=[f"{AGENT_NAME}-{get_timestamp_string()}-{idx}" for idx in range(len(state["inputs"]))]
    requests=[
        Request(
            custom_id=ids[idx],
            params=MessageCreateParamsNonStreaming(
                model=config.get("configurable", {}).get("model", "claude-3-haiku-20240307"),
                max_tokens= "**********"
                messages=[{"role":"user", "content": input}],
            )
        ) for idx, input in enumerate(state["inputs"])
    ]
    batch=client.messages.batches.create(
        requests=requests
    )
    return {"batch": batch, "ids": ids}

def reorder_results(ids, results):
  ids_map={id: idx for idx, id in enumerate(ids)}
  ordered=list(range(0, len(ids)))
  for res in results:
    ordered[ids_map[res.custom_id]]=res
  return ordered
def get_results(state: State):
  results = client.messages.batches.results(
        state["batch"].id
      )
  # We don't have a guarantee for order
  reordered_results=reorder_results(state["ids"], results)
  # TODO: handler errors as well
  parsed_results=[result.result for result in reordered_results]
  return {"results": parsed_results }


workflow = StateGraph(State)
workflow.add_node(batch_runner)
workflow.add_node(get_results)
workflow.add_conditional_edges(START, decide_running)

checkpointer = memory
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "2"}}
# https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#supported-models
graph.invoke({"inputs": ["bonjour", "hello"]}, config, checkpoint_during=True)

os.listdir(CHECKPOINT_FOLDER)uring=True)

os.listdir(CHECKPOINT_FOLDER)