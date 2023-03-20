#date: 2023-03-20T16:51:02Z
#url: https://api.github.com/gists/5571b601c114d4a24c0e8d2c3e9e7e5d
#owner: https://api.github.com/users/Shelob9

from openai import Model, Engine
from gpt_index.composability import ComposableGraph
from gpt_index.readers import GithubRepositoryReader
from gpt_index import download_loader,GPTTreeIndex,Document,MockLLMPredictor,GPTListIndex,PromptHelper,GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from dotenv import load_dotenv
from pathlib import Path
from prepare import unmark

load_dotenv()
import os
class Indexer :
    def __init__(self):
        # set maximum input size
        max_input_size = 512
        # set number of output tokens
        num_output = 256
        # set maximum chunk overlap
        max_chunk_overlap = 20
        self.prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
        self.mock_predictor = "**********"=num_output)
        self.llm_predictor = LLMPredictor(llm=Model(temperature=.3, model_name="text-ada-001"))
        self.indexes = []
    def index_documents(self,documents,save,summarize=False):
        index = GPTSimpleVectorIndex(
            documents, llm_predictor=self.llm_predictor, prompt_helper=self.prompt_helper
        )
        if summarize :
            summary = index.query("What is a summary of this document?")
            index.set_text(str(summary))
        if save:
            index.save_to_disk(save)
        self.indexes.append(index)
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"e "**********"s "**********"t "**********"i "**********"m "**********"a "**********"t "**********"e "**********"_ "**********"i "**********"n "**********"d "**********"e "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"d "**********"o "**********"c "**********"u "**********"m "**********"e "**********"n "**********"t "**********"s "**********") "**********": "**********"
        index = GPTTreeIndex(documents, llm_predictor=self.mock_predictor)
        # get number of tokens used
        print(self.mock_predictor.last_token_usage)

    def index_youtube(self,location,save):
        YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")

        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=[location])
        self.index_documents(documents,save,False)

    def index_pdf(self,location,save):
        PDFReader = download_loader("PDFReader")

        loader = PDFReader()
        documents = loader.load_data(file=Path(location))
        self.index_documents(documents,save,False)
    def index_markdown(self,location,save):
        documents =[]
        # itterate directory location
        for path in os.listdir(location):
            # check if current path is a file
            if os.path.isfile(os.path.join(location, path)):
                f = open(file=Path(location+'/'+path))
                f = unmark(str(f.read()))
                # load file to documents

                documents.append(Document(f))
        self.index_documents(documents,save,False)
    def make_graph(self,save_graph):
        list_index = GPTListIndex(self.indexes)
        if save_graph:
            graph = ComposableGraph.build_from_index(list_index)
            graph.save_to_disk(save_graph)
    def load_index(self,location):
        index = GPTSimpleVectorIndex.load_from_disk(location)
        self.indexes.append(index)


class Github :
    def __init__(self,owner,repo):
        self.reader = GithubRepositoryReader(owner,repo)
        self.documents = self.reader.load_data(branch="main")

class Queryer :
    def __init__(self,location):
        self.graph = ComposableGraph.load_from_disk(location)
        self.response = ""
    def query(self,query):
        query_configs = [
            {
                "index_struct_type": "tree",
                "query_mode": "default",
                "query_kwargs": {
                    "child_branch_factor": 2
                }
            },
            {
                "index_struct_type": "keyword_table",
                "query_mode": "simple",
                "query_kwargs": {}
            },
        ]
        self.response = self.graph.query(query, query_configs=query_configs)
