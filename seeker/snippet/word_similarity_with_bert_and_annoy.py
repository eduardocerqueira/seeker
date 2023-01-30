#date: 2023-01-30T17:03:33Z
#url: https://api.github.com/gists/26d0f5d74b3be45154fa3cb06cb87d4c
#owner: https://api.github.com/users/angelosalatino

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:31:14 2023

@author: aas358
"""

import torch
from transformers import AutoTokenizer, AutoModel
from annoy import AnnoyIndex



class ContextualEmbeddings:
    """Class for contextual embeddings extraction via BERT"""

    def __init__(self):
        # Use last four layers by default
        self.tokenizer = "**********"
        self.model = AutoModel.from_pretrained("bert-base-cased", output_hidden_states=True)

    def get_embeddings_of_each_word(self, sentence:str)-> list:
        """
        Extracts the embeddings of all the individual words into a sentence.
        As complex words may consist of multiple tokens, the function will merge
        the different embeddings.

        Parameters
        ----------
        tokenizer : "**********"
        model : AutoModel
        sentence : str
            The sentence to which extract embeddings.

        Returns
        -------
        list
            The embeddings. Each item in the list is the embedding of each word.

        """
        layers = [-4, -3, -2, -1]

        encoded = "**********"="pt")

        # get all token idxs belonging to the words of interest

        number_of_total_words = len(set(filter(lambda e: isinstance(e, int), encoded.word_ids())))

        token_ids_words = "**********"
        words = []
        for word_id in range(number_of_total_words):
            token_ids_words.append(range(encoded.word_to_tokens(word_id).start,encoded.word_to_tokens(word_id).end))
            words.append(sentence[encoded.word_to_chars(word_id).start:encoded.word_to_chars(word_id).end])

        with torch.no_grad():
            output = self.model(**encoded)

        # Get all hidden states
        states = output.hidden_states

        # Stack and sum all requested layers
        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
        # Only select the tokens that constitute the requested word
        word_embeddngs = []
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********"_ "**********"w "**********"o "**********"r "**********"d "**********"  "**********"i "**********"n "**********"  "**********"z "**********"i "**********"p "**********"( "**********"w "**********"o "**********"r "**********"d "**********"s "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"s "**********"_ "**********"w "**********"o "**********"r "**********"d "**********"s "**********") "**********": "**********"
            word_embeddngs.append((word,output[token_ids_word].mean(dim= "**********"

        return word_embeddngs



if __name__ == "__main__":
    
    LEN_EMBEDDINGS = 768 #length of BERT Embeddings
    sentence = """Research publishing companies need to constantly monitor and compare scientific journals and conferences in order to inform critical business and editorial decisions. Semantic Web and Knowledge Graph technologies are natural solutions since they allow these companies to integrate, represent, and analyse a large quantity of information from heterogeneous sources. In this paper, we present the AIDA Dashboard 2.0, an innovative system developed in collaboration with Springer Nature to analyse and compare scientific venues, now also available to the public. This tool builds on a knowledge graph which includes over 1.5B RDF triples and was produced by integrating information about 25M research articles from Microsoft Academic Graph, Dimensions, DBpedia, GRID, CSO, and INDUSO. It can produce sophisticated analytics and rankings that are not available in alternative systems. We discuss the advantages of this solution for the Springer Nature editorial process and present a user study involving 5 editors and 5 researchers, which yielded excellent results in terms of quality of the analytics and usability."""
    
    ce = ContextualEmbeddings()
    word_embeddings = ce.get_embeddings_of_each_word(sentence)
    
    annoy = AnnoyIndex(LEN_EMBEDDINGS, 'angular')
    for idx, word in enumerate(word_embeddings):
        annoy.add_item(idx, word[1].numpy())
    
    annoy.build(10)
    
    
    sentence_test = """Scientific conferences are essential for developing active research communities, promoting the cross-pollination of ideas and technologies, bridging between academia and industry, and disseminating new findings. Analyzing and monitoring scientific conferences is thus crucial for all users who need to take informed decisions in this space. However, scholarly search engines and bibliometric applications only provide a limited set of analytics for assessing research conferences, preventing us from performing a comprehensive analysis of these events. In this paper, we introduce the AIDA Dashboard, a novel web application, developed in collaboration with Springer Nature, for analyzing and comparing scientific conferences. This tool introduces three major new features: 1) it enables users to easily compare conferences within specific fields (e.g., Digital Libraries) and time-frames (e.g., the last five years); 2) it characterises conferences according to a 14K research topics from the Computer Science Ontology (CSO); and 3) it provides several functionalities for assessing the involvement of commercial organizations, including the ability to characterize industrial contributions according to 66 industrial sectors (e.g., automotive, financial, energy, electronics) from the Industrial Sectors Ontology (INDUSO). We evaluated the AIDA Dashboard by performing both a quantitative evaluation and a user study, obtaining excellent results in terms of quality of the analytics and usability."""
    word_embeddings_test = ce.get_embeddings_of_each_word(sentence_test)
    
    idx_word_to_test = 1
    
    print(f"I will test what are the most similar words to {word_embeddings_test[idx_word_to_test][0]}")
    indices, dists = annoy.get_nns_by_vector(word_embeddings_test[idx_word_to_test][1].numpy(), 
                                             5, 
                                             include_distances=True)
    
    for index, distance in zip(indices, dists):
        print(f"{word_embeddings_test[idx_word_to_test][0]} is similar to {word_embeddings[index][0]}, with similarity {1-distance}")
    
    
    
    
