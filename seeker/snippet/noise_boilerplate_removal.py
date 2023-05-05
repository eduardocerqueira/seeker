#date: 2023-05-05T16:50:17Z
#url: https://api.github.com/gists/de2f706c183308e05726169508ae1cb2
#owner: https://api.github.com/users/rjrahul24

import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

def remove_noise_boilerplate(input_text, min_cluster_size=2, num_clusters=5, max_noise_ratio=0.3):
    
    # Sentence split: To identify boilerplate/noise we will first need to separate sentences to find similarity
    sentences = re.split('\. |\? |\! |\n', input_text)
    
    # Convert sentences to a matrix of word embeddings
    embeddings_matrix = text_vectorize(sentences)
    
    # KMeans Clustering: Cluster the sentences to bring similar embeddings together 
    kmeans_model = KMeans(n_clusters=num_clusters)
    kmeans_model.fit(embeddings_matrix)
    model_labels = kmeans_model.labels_
    model_centroids = kmeans_model.cluster_centers_
    
    # Individual cluster size
    cluster_sizes = np.bincount(model_labels)
    
    # Identify clusters with noise and boilerplate language
    is_noise = np.zeros(num_clusters, dtype=bool)
    for i, centroid in enumerate(model_centroids):
        if cluster_sizes[i] < min_cluster_size:
            # We should ignore clusters with fewer sentences than min_cluster_size threshold
            continue
        distances = np.linalg.norm(embeddings_matrix[model_labels == i] - centroid, axis=1)
        median_distance = np.median(distances)
        if np.count_nonzero(distances > median_distance) / cluster_sizes[i] > max_noise_ratio:
            is_noise[i] = True
    
    # Remove: Sentences that are in the noise bucket, we remove them (boilerplate)
    filtered_sentences = []
    for i, sentence in enumerate(sentences):
        if not is_noise[model_labels[i]]:
            filtered_sentences.append(sentence)
    
    # Bring the sentence together
    filtered_text = ' '.join(filtered_sentences)
    
    return filtered_text

def text_vectorize(input_text):
    
    # Instantiate the CountVectorizer object
    vectorizer = CountVectorizer()
    
    # Use vectorizer.fit to transform the text into a matrix of word counts
    counts_matrix = vectorizer.fit_transform(input_text)
    
    # Convert to a dense matrix
    dense_matrix = counts_matrix.todense()
    
    # Return the dense matrix as a numpy array
    return np.array(dense_matrix)