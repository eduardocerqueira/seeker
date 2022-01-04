#date: 2022-01-04T17:11:47Z
#url: https://api.github.com/gists/c67fd3ef3f43e842e53cc8e92ffd9cef
#owner: https://api.github.com/users/automationdream

from typing import List, Optional, Any
import json
from glob import glob

import ray
from machine_learning_tools.ocr import convert_abbyy_to_words, get_dict_from_abbyy_xml

from pydantic import BaseModel
from ray import serve

from invoice_extractor.common.load_models import ModelStack, ModelConfig


# TODO add validations
# TODO change embedding_file to something that can associate to an embedding service

class ExtractorModelConfig(BaseModel):
    network_class: str
    model_file: str
    post_processor: str
    embedding_file: str
    transfer_learning: bool = False
    shift_overlaps: bool = False


class ModelEnsembleConfig(BaseModel):
    model_configs: List[ModelConfig]
    entity_merge_probability_threshold: float
    document_class: Any = None
    models_schema: Optional[dict] = None
    page_chunk_size: int = 1
    accounting_models: List = []


if __name__ == "__main__":
    def get_docs_from_xml(path_to_xmls: str = "../tests/full_pipeline/invoice_osram_my/*.xml") -> List:
        return list(map(
            convert_abbyy_to_words,
            map(
                get_dict_from_abbyy_xml,
                [open(file, "rb").read() for file in glob(path_to_xmls, recursive=True)]
            )
        ))


    with open('../models/information_extractor_oracle_receipt/schema.json', 'rb') as f:
        hy_schema = f.read()

    a = ModelStack(
        model_configs=[
            ModelConfig(
                network_class="WordResNet",
                model_file="57679a5b-46a6-4773-b029-9dddee3b1700-2.dill",
                model_file_dir=".",
                embedding_file="embedding-1-100.bin",
                post_processor="InvoiceOsramMy",
                transfer_learning=False
            ),
            ModelConfig(
                network_class="WordResNet",
                model_file="57679a5b-46a6-4773-b029-9dddee3b1700-2.dill",
                model_file_dir=".",
                embedding_file="embedding-1-100.bin",
                post_processor="InvoiceOsramOG",
                transfer_learning=False
            ),
        ],
        page_chunk_size=1,
        entity_merge_probability_threshold=0.03,
        models_schema=json.loads(hy_schema),
    )

    docs = get_docs_from_xml()
    resp = a.run_extraction_models(ocr_doc=docs[0]).to_dict()

