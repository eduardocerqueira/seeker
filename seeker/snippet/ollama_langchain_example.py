#date: 2024-01-22T16:59:19Z
#url: https://api.github.com/gists/329cad8e8cce7c0798b2fba9f134f248
#owner: https://api.github.com/users/guidoman

# Adapted from: https://js.langchain.com/docs/use_cases/question_answering/local_retrieval_qa

# Requirements:
# pip install langchain
# pip install beautifulsoup4
# pip install sentence-transformers
# pip install faiss-cpu


from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

loader = WebBaseLoader(
    "https://lilianweng.github.io/posts/2023-06-23-agent/"
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_overlap=0, chunk_size=500)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(split_docs, embeddings)
retriever = vector_store.as_retriever()

model = Ollama(
    base_url="http://localhost:11434",
    model="llama2")

prompt_template = PromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

question = "What are the approaches to Task Decomposition?"
result = qa_chain({"query": question})
print("Result:")
print(result["result"])
print("Source document:")
print(result["source_documents"][0])
