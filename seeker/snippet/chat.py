#date: 2023-06-02T16:43:22Z
#url: https://api.github.com/gists/05c4c95e45994e454e366085356de39d
#owner: https://api.github.com/users/powerwlsl

import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.chains import LLMChain
import sys

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment="us-west1-gcp-free"
    )
    

def run_llm(query:str):
    # Load the documents
    loader = TextLoader("/Users/hyejinahn/ice_breaker/chat.txt")
    document = loader.load()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n", "\r\n"]

    )

    # Split the documents into chunks
    docs = text_splitter.split_documents(document)

    # Initialize the embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize the vectorstore

    # index = pinecone.Index("chat-index")
    # index.delete(deleteAll='true')

    # docsearch = Pinecone.from_documents(
    #     docs,
    #     index_name="chat-index",
    #     embedding=embeddings,
    # )

    # Getting the existing index
    docsearch = Pinecone.from_existing_index('chat-index', embeddings)
    
    # Initialize the chat model
    chat = ChatOpenAI(
        temperature=0,
        verbose=True,
    )

    # Initialize the QA model
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        # return_source_documents=True,
        verbose=True,
    )

    template = """
    given the document, I want you to act and answer like 언니
    Hyejin Ahn 안혜진: {query} \n
    언니: 
    """

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=template,
    )
    
    chain = LLMChain(
        llm=chat,
        prompt=prompt_template
    )

    return chain.run(query)

if __name__ == "__main__":
    var = input("You: ")

    print("언니: ", run_llm(var))
    

