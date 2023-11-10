#date: 2023-11-10T17:09:09Z
#url: https://api.github.com/gists/583ee9f8d2a21a562f42535da47cee0d
#owner: https://api.github.com/users/thoraxe

# ...
# ...

class DocsSummarizer:
    def __init__(self):
        self.logger = OLSLogger("docs_summarizer").logger

    def summarize(self, conversation, query, **kwargs):
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = DEFAULT_MODEL

        if "verbose" in kwargs:
            if kwargs["verbose"] == 'True' or kwargs["verbose"] == 'true':
                verbose = True
            else:
                verbose = False
        else:
            verbose = False

        # make llama index show the prompting
        if verbose == True:
            llama_index.set_global_handler("simple")

        settings_string = f"conversation: {conversation}, query: {query},model: {model}, verbose: {verbose}"
        self.logger.info(
            conversation
            + " call settings: "
            + settings_string
        )

        summarization_template_str = """
The following context contains several pieces of documentation. Please summarize the context for the user.
Documentation context:
{context_str}

Summary:

"""
        summarization_template = PromptTemplate(
            summarization_template_str
        )

        self.logger.info(conversation + " Getting sevice context")
        self.logger.info(conversation + " using model: " + model)

        ## check if we are using remote embeddings via env
        tei_embedding_url = os.getenv("TEI_SERVER_URL", None)
        
        if tei_embedding_url != None:
            service_context = get_watsonx_context(model=model, 
                                              tei_embedding_model='BAAI/bge-base-en-v1.5',
                                              url=tei_embedding_url)
        else:
          service_context = get_watsonx_context(model=model)


        storage_context = StorageContext.from_defaults(persist_dir="vector-db/ocp-product-docs")
        self.logger.info(conversation + " Setting up index")
        index = load_index_from_storage(
            storage_context=storage_context,
            index_id="product",
            service_context=service_context,
        )

        self.logger.info(conversation + " Setting up query engine")
        query_engine = index.as_query_engine(
            text_qa_template=summarization_template,
            verbose=verbose,
            streaming=False, similarity_top_k=1
        )

        # TODO: figure out how to log the full query sent to the query engine in a better way

        self.logger.info(conversation + " Submitting summarization query")
        summary = query_engine.query(query)

        referenced_documents = ""
        for source_node in summary.source_nodes:
            # print(source_node.node.metadata['file_name'])
            referenced_documents += source_node.node.metadata["file_name"] + "\n"

        self.logger.info(conversation + " Summary response: " + str(summary))
        for line in referenced_documents.splitlines():
            self.logger.info(conversation + " Referenced documents: " + line)

        return str(summary), referenced_documents