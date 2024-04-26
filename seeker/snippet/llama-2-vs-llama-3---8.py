#date: 2024-04-26T16:50:19Z
#url: https://api.github.com/gists/c56f839e50ac1aea24591c09c7094116
#owner: https://api.github.com/users/BobMerkus

from langchain_community.chat_models.ollama import ChatOllama
model_arguments = {
    'temperature': 0.5, # this is the temperature parameter, higher means more randomness
    'max_tokens': "**********"
    'top_k': "**********"
    'top_p': "**********"
    'verbose': False, # tell langchain to not print debug information
    'streaming': True, # tell langchain we want to stream
    'device': 'cuda:0' # tell langchain to use the GPU
}
llama_2 = ChatOllama(name="llama2", model='llama2', **model_arguments)
llama_3 = ChatOllama(name="llama3", model='llama3', **model_arguments)we want to stream
    'device': 'cuda:0' # tell langchain to use the GPU
}
llama_2 = ChatOllama(name="llama2", model='llama2', **model_arguments)
llama_3 = ChatOllama(name="llama3", model='llama3', **model_arguments)