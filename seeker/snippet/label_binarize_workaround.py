#date: 2022-07-19T16:52:07Z
#url: https://api.github.com/gists/d11554639d45e8c06e93568b0f17788c
#owner: https://api.github.com/users/AlexHenderson

import numpy as np
from sklearn.preprocessing import label_binarize


def my_label_binarizer(labels):
    unique_labels = list(dict.fromkeys(labels))

    binary_labels = None

    if len(unique_labels) == 1:
        binary_labels = np.ones_like(labels, dtype=bool)

    elif len(unique_labels) == 2:
        binary_labels = label_binarize(labels, classes=unique_labels)
        binary_labels = binary_labels > 0  # convert to boolean
        binary_labels = np.concatenate((~binary_labels, binary_labels), axis=1)

    elif len(unique_labels) > 2:
        binary_labels = label_binarize(labels, classes=unique_labels)
        binary_labels = binary_labels > 0  # convert to boolean

    return binary_labels, unique_labels

  
# Multi-class example 
# labels3 = ['dog', 'cat', 'dog', 'cat', 'rabbit']
# binary_labels3, unique_labels3 = my_label_binarizer(labels3)
# print(f"Original labels: {labels3}")
# print(f"Unique labels: {unique_labels3}")
# print(f"Binarized labels: ")
# print(binary_labels3)
# print(f"shape = {binary_labels3.shape}")

# Output...
# Original labels: ['dog', 'cat', 'dog', 'cat', 'rabbit']
# Unique labels: ['dog', 'cat', 'rabbit']
# Binarized labels: 
# [[ True False False]
#  [False  True False]
#  [ True False False]
#  [False  True False]
#  [False False  True]]
# shape = (5, 3)

  
# Two class example
# labels2 = ['dog', 'cat', 'dog', 'cat']
# binary_labels2, unique_labels2 = my_label_binarizer(labels2)
# print(f"Original labels: {labels2}")
# print(f"Unique labels: {unique_labels2}")
# print(f"Binarized labels: ")
# print(binary_labels2)
# print(f"shape = {binary_labels2.shape}")

# Output...
# Original labels: ['dog', 'cat', 'dog', 'cat']
# Unique labels: ['dog', 'cat']
# Binarized labels: 
# [[ True False]
#  [False  True]
#  [ True False]
#  [False  True]]
# shape = (4, 2)


# Single class example for completeness
# labels1 = ['dog', 'dog', 'dog']
# binary_labels1, unique_labels1 = my_label_binarizer(labels1)
# print(f"Original labels: {labels1}")
# print(f"Unique labels: {unique_labels1}")
# print(f"Binarized labels: ")
# print(binary_labels1)
# print(f"shape = {binary_labels1.shape}")

# Output...
# Original labels: ['dog', 'dog', 'dog']
# Unique labels: ['dog']
# Binarized labels: 
# [ True  True  True]
# shape = (3,)
