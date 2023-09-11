#date: 2023-09-11T17:00:59Z
#url: https://api.github.com/gists/cdd6a50c1790be9ae1950a6fb73471ec
#owner: https://api.github.com/users/dwsmart

def Jaccard_Similarity(doc1, doc2): 
    
    # List the unique words in a document
    words_doc1 = set(doc1.lower().split()) 
    words_doc2 = set(doc2.lower().split())
    
    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)

doc_1 = ''' page one text here'''
doc_2 = ''' page two text here'''


sim = Jaccard_Similarity(doc_1,doc_2)
print("Jaccard similarity between doc_1 & doc_2 is", sim)