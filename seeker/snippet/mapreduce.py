#date: 2024-08-15T17:06:34Z
#url: https://api.github.com/gists/06ba2e58fe68d22d32be697518c72682
#owner: https://api.github.com/users/HorseCheng

import gc
import os
import time

import natsort
import torch
import whisper
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import SRTLoader
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_text_splitters import (CharacterTextSplitter,
                                      RecursiveCharacterTextSplitter)

file_name = "xxxxxxxxxxxxxxxxxxxxxx"
loader = SRTLoader(file_name+".srt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                               chunk_overlap=10)
langchain_splits = text_splitter.split_documents(docs)
# print(langchain_splits)
# print(len(langchain_splits))
# exit()
llm = Ollama(
    # model="llama3.1",num_ctx=3000
    model="gemma2",num_ctx=3000
)

prompt_template = "以下為字幕檔: {text}\n=========\n 請根據以上字幕生成摘要並確保核心內容"
# query = f"以下為字幕檔: {docs[0]}\n=========\n 請根據以上字幕生成 詳盡的summary報告"
# query = f"以下為字幕檔: {docs[0]}\n=========\n 請根據以上字幕。 針對每一小段落，提供詳盡的精華、獨特觀點與重點，包含key 舉例 summary"
# print(f"{file_name} 字幕摘要: ")
# for chunks in llm.stream(query):
#     print(chunks, end="")

prompt = PromptTemplate.from_template(prompt_template)
chain = load_summarize_chain(llm=llm,
                             prompt=prompt,
                             chain_type="stuff")
print(chain.invoke(docs)["output_text"])

print("====================")

reduce_prompt = ChatPromptTemplate.from_messages(
    [("system", "以下為文件內容: {text}\n=========\n 請將這些內容進行總結且保持核心內容")]
)

map_prompt = ChatPromptTemplate.from_messages(
    [("system", "以下是一組字幕檔串列：\n {text}\n======\n 請根據以上字幕串列生成摘要並確保核心內容")]
)

# chain = load_summarize_chain(llm=llm,
#                              combine_prompt=reduce_prompt,
#                              map_prompt=map_prompt,
#                              chain_type="map_reduce",verbose=True)
# print(chain.invoke(langchain_splits)['output_text'])

response_history = []
for i in langchain_splits:
    print("=====================")
    print("Text:", i.page_content)
    print("-----------------")
    initial_prompt = map_prompt.format(text=i.page_content)
    response = llm.invoke(initial_prompt)
    print(response)
    response_history.append(response)

print("======summary===============")
summary = "\n".join(response_history)
final_prompt = reduce_prompt.format(text=summary)
response = llm.invoke(final_prompt)
print(response)
