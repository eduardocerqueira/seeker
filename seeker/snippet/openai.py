#date: 2023-08-24T16:31:38Z
#url: https://api.github.com/gists/6184232e4f4ddeb6ca2070aceada5c69
#owner: https://api.github.com/users/jerryjliu

# For regular OpenAI
import os
import json
import openai
from llama_index.llms import OpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from llama_index import LangchainEmbedding
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
)

from llama_index import set_global_service_context

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
# from llama_index.llms import LangChainLLM

import logging
import sys

from IPython.display import Markdown, display

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")


llm = OpenAI(model="gpt-3.5-turbo")

# You need to deploy your own embedding model as well as your own chat completion model
# embedding_llm = LangchainEmbedding(
#     OpenAIEmbeddings(
#         model="text-embedding-ada-002",
#         deployment=os.environ["OPENAI_EMBEDDING_ENGINE"],
#         openai_api_key=openai.api_key,
#         openai_api_base=openai.api_base,
#         openai_api_type=openai.api_type,
#         openai_api_version=openai.api_version,
#     ),
#     embed_batch_size=1,
# )
service_context = ServiceContext.from_defaults(
    llm=llm,
    # embed_model=embedding_llm,
)

# SET Global Service Context
set_global_service_context(service_context)