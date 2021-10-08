#date: 2021-10-08T17:15:24Z
#url: https://api.github.com/gists/f61cf9b5d6a77e156f213d6d7082a2b0
#owner: https://api.github.com/users/hugozanini

import json

from torch import nn
from transformers import BertModel

with open("config.json") as json_file:
    config = json.load(json_file)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        #TODO: O que é esse super?
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config["BERT_MODEL"],
                                              return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.drop(pooled_output)
        return self.out(output)