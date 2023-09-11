#date: 2023-09-11T17:00:26Z
#url: https://api.github.com/gists/79fb387d1fcb9b4c84c0f2f6c45e63c8
#owner: https://api.github.com/users/joshreini1

vector_store = MilvusVectorStore(index_params={
        "index_type": index_param,
        "metric_type": "L2"
        },
        search_params={"nprobe": 20},
        overwrite=True)
llm = OpenAI(model="gpt-3.5-turbo")
storage_context = StorageContext.from_defaults(vector_store = vector_store)
service_context = ServiceContext.from_defaults(embed_model = embed_model, llm = llm, chunk_size=chunk_size)
index = VectorStoreIndex.from_documents(wiki_docs,
        service_context=service_context,
        storage_context=storage_context)