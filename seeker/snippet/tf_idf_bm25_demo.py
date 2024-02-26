#date: 2024-02-26T17:03:24Z
#url: https://api.github.com/gists/5a84ff067db4dcbf7a3db153e13a0d1f
#owner: https://api.github.com/users/Dharin-shah

import math
from collections import Counter

# Sample documents
documents = [
    'the sky is blue',
    'the sun is bright',
    'the sun in the sky is bright',
    'the shining sun, the bright sun is amazing',
     'northern lights in iceland are breathtaking',
    'sweden is famous for its northern lights',
    'norway offers stunning views of the northern lights',
    'huskies are common in iceland and norway',
    'iceland is known for its glaciers and volcanoes',
    'sweden has a rich history and beautiful landscapes',
    'norway is famous for fjords and mountainous terrain',
    'the northern lights are a natural phenomenon',
    'huskies are sled dogs used in snowy regions',
    'iceland and norway are popular destinations for viewing the northern lights',
    'swedish cuisine is unique and diverse',
    'the norwegian fjords are a must-see natural wonder',
    'husky tours are a popular activity in iceland',
    'sweden\'s archipelagos offer stunning natural beauty',
    'norway\'s coastal areas are known for their fishing industry',
    'the aurora borealis is another name for the northern lights',
    'huskies have thick fur coats to protect them from cold',
    'iceland\'s geothermal activity creates hot springs and geysers',
    'sweden is home to the Nobel Prize',
    'norwegian wood is a famous novel and song',
    'pizza tours in Italy offer a unique culinary experience',
    'food tours in Paris introduce a variety of French cuisine',
    'deep-sea fishing tours are popular in Norway',
    'exploring the best pizza joints in New York City',
    'Japanese food tours provide an in-depth look at sushi making',
    'fishing in the clear waters of Iceland is a serene experience',
    'the art of making Neapolitan pizza',
    'culinary tours in India reveal the diversity of Indian cuisine',
    'Norwegian salmon fishing is recognized worldwide',
    'street food tours in Bangkok are a delight for foodies',
    'the tradition of pizza making in Naples',
    'seafood and fishing culture in coastal regions',
    'discover the best pizza toppings on a Rome food tour',
    'authentic sushi tours in Tokyo',
    'the experience of ice fishing in cold regions'
]

def compute_tf(word_dict, doc_length):
    tf_dict = {}
    for word, count in word_dict.items():
        tf_dict[word] = count / float(doc_length)
    return tf_dict

def compute_idf(doc_freq, total_docs):
    idf_dict = {}
    for word, count in doc_freq.items():
        idf_dict[word] = math.log(total_docs / float(count))
    return idf_dict

def compute_tfidf(tfs, idfs):
    tfidf = {}
    for word, tf_val in tfs.items():
        tfidf[word] = tf_val * idfs.get(word, 0)
    return tfidf

def preprocess_and_compute_tfidf(documents):
    doc_freq = Counter()
    word_counts = [Counter(doc.lower().split()) for doc in documents]
    all_tfs = []
    for word_count in word_counts:
        doc_freq.update(word_count.keys())
        all_tfs.append(compute_tf(word_count, sum(word_count.values())))
    idfs = compute_idf(doc_freq, len(documents))
    tfidfs = [compute_tfidf(tfs, idfs) for tfs in all_tfs]
    return tfidfs, idfs

tfidfs, idfs = preprocess_and_compute_tfidf(documents)

k1 = 1.5
b = 0.75

def compute_bm25(idf, tf, doc_length, avg_doc_length):
    return idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length))))

def bm25_scores(term, documents, tfidfs, idfs, avg_doc_length):
    scores = []
    for doc, tfidf in zip(documents, tfidfs):
        doc_length = len(doc.split())
        tf = tfidf.get(term, 0)
        scores.append(compute_bm25(idfs.get(term, 0), tf, doc_length, avg_doc_length))
    return scores


def search_and_display(phrase, documents, tfidfs, idfs, avg_doc_length):
    terms = phrase.lower().split()  

    print(f"Searching for phrase: '{phrase}'\n")
    
    combined_scores = []

    for term in terms:
        term_bm25_scores = bm25_scores(term, documents, tfidfs, idfs, avg_doc_length)
        
        if combined_scores:
            combined_scores = [combined + term_score for combined, term_score in zip(combined_scores, term_bm25_scores)]
        else:
            combined_scores = term_bm25_scores
    
    matching_documents = sorted([(i, score) for i, score in enumerate(combined_scores, start=1) if score > 0], key=lambda x: x[1], reverse=True)

    if matching_documents:
        print("Matching Documents and Scores (Ordered by Score):")
        for doc_index, score in matching_documents:
            print(f"Document {doc_index} Content: '{documents[doc_index - 1]}'")
            print(f"Score: {score:.4f}\n")
    else:
        print(f"No documents matching the phrase '{phrase}' with score > 0.")



def explain_logarithm(term, total_docs, doc_freq):
    idf = math.log(total_docs / doc_freq)
    print(f"\nExplaining IDF for '{term}':")
    print(f"Total Documents: {total_docs}, Documents with '{term}': {doc_freq}")
    print(f"IDF = log(Total Documents / Documents with '{term}') = log({total_docs} / {doc_freq}) = {idf:.4f}")
    print("The logarithm scales down the IDF value, making the term's frequency across documents less dominant in the final score.\n")


def display_idf_matrix(idfs):
    print("\nIDF Scores Matrix:")
    
    sorted_terms = sorted(idfs.items(), key=lambda x: x[1])
    
    print(f"{'Term':<15}{'IDF Score'}")
    
    for term, score in sorted_terms:
        print(f"{term:<15}{score:.4f}")

def display_scores_matrix(documents, tfidfs, idfs):
    for doc_index, (doc, tfidf_scores) in enumerate(zip(documents, tfidfs), start=1):
        print(f"\nDocument {doc_index} Scores Matrix:")
        print(f"{'Term':<15}{'TF':<10}{'IDF':<10}{'TF-IDF':<10}")
        
        terms = doc.lower().split()
        term_counts = Counter(terms)
        doc_length = len(terms)
        
        for term in sorted(term_counts.keys()): 
            tf = term_counts[term] / doc_length
            idf = idfs.get(term, 0)
            tf_idf = tfidf_scores.get(term, 0)
            print(f"{term:<15}{tf:<10.4f}{idf:<10.4f}{tf_idf:<10.4f}")

def display_overall_scores_matrix(tfidfs, idfs, total_docs):
    print("\nOverall Scores Matrix:")
    print(f"{'Term':<20}{'Cumulative TF':<20}{'IDF':<20}{'Average TF-IDF':<20}")
    
    cumulative_tf = {term: sum(tfidf.get(term, 0) for tfidf in tfidfs) for term in idfs.keys()}
    
    avg_tfidf = {term: cumulative_tf[term] / total_docs for term in idfs.keys()}
    
    sorted_terms = sorted(idfs.keys(), key=lambda term: idfs[term])
    
    for term in sorted_terms:
        print(f"{term:<20}{cumulative_tf[term]:<20.4f}{idfs[term]:<20.4f}{avg_tfidf[term]:<20.4f}")





if __name__ == "__main__":

    # Preprocess documents and compute TF-IDF
    tfidfs, idfs = preprocess_and_compute_tfidf(documents)
    doc_freq = Counter([word for tfidf in tfidfs for word in tfidf])

    # Calculate average document length
    avg_doc_length = sum(len(doc.split()) for doc in documents) / len(documents)
    
    term_to_search = input("Enter a term to search for: ")
    search_and_display(term_to_search, documents, tfidfs, idfs, avg_doc_length)
    
    # Display IDF matrix
    display_overall_scores_matrix(tfidfs, idfs, len(documents))