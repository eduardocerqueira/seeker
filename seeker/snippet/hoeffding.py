#date: 2022-02-21T16:59:26Z
#url: https://api.github.com/gists/741066b42bbdad93fe7560af0ee246d0
#owner: https://api.github.com/users/xkianteb

import numpy as np
import pandas as pd
import numpy as np
import multiprocessing as mp
import scipy.stats  as stats

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.colors import ListedColormap

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# HoeffdingBounds
def getHoeffdingBound(n, epsilon):
    return 2.0 * np.exp(-2 * n * epsilon * epsilon)

# Gets the difference between the sample mean and the real expectation
def getSampleDelta(n, p):
    r = stats.binom.rvs(n, p)
    realMean = r/(1.0 * n)
    return abs(realMean-p)

def getRealBound(n, p, epsilon):
    trials = 100
    samples = [ getSampleDelta(n, p) for _ in range(trials) ]
    events = [x for x in samples if x > epsilon]
    return len(events)/(1.0 * len(samples))

# Trains a model with Naive Bayes and returns the train and test accuracy
def trail(n_samples):
    rng = np.random.RandomState(n_samples[0])
    n_samples = n_samples[1]

    X, y = make_classification(n_features=5, n_redundant=0, n_informative=3, random_state=1, n_clusters_per_class=2, n_samples=3000)
    clf = GaussianNB()

    X_train = X[:2000][:]
    y_train = y[:2000][:]

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    (X_train, y_train) = unison_shuffled_copies(X_train, y_train)

    X_train = X_train[:n_samples]
    y_train = y_train[:n_samples]

    X_test = X[2000:][:]
    y_test = y[2000:][:]

    clf.fit(X_train, y_train)

    train = clf.score(X_train, y_train)
    test = clf.score(X_test, y_test)
    return {'train':train, 'test':test}
    
if __name__=="__main__":
    # Hoeffdings Inequality with binomial discrete random variables
    prob = 0.9
    eps = 0.25

    narray = np.arange(2, 100, 1)
    barray = [getHoeffdingBound(n, eps) for n in narray]
    rarray = [getRealBound(n, prob, eps) for n in narray]

    plt.scatter(narray, barray, c='b')#, marker='x', label='1')
    plt.scatter(narray, rarray, c='r')#, marker='s', label='-1')
    plt.savefig('Hoeffdings_binomial.png')
    plt.clf()

    # Hoeffdings Inequality with Naive Bayes Classification
    eps = 0.15

    rarray = []
    avg_E_in = []
    avg_E_out = []
    narray = []

    for n_samples in range(2, 150, 1):
        corpus = {id: (n_samples) for id in range(500)}
        with mp.Pool() as pool:
            results = pool.map(trail, corpus.items())

        E_in = [results[x]['train'] for x in range(500)]
        E_out = [results[x]['test'] for x in range(500)]

        samples = abs(np.array(E_in) - np.array(E_out))
        bad_events = [x for x in samples if x > eps]
        prob_bad_events = len(bad_events)/(1.0 * len(samples))

        rarray.append(prob_bad_events)
        narray.append(n_samples)
        avg_E_in.append(np.mean(E_in))
        avg_E_out.append(np.mean(E_out))

    barray = [getHoeffdingBound(n, eps) for n in narray]

    plt.scatter(narray, barray, c='b')
    plt.scatter(narray, rarray, c='r')
    plt.savefig('Hoeffdings_Naive_Bayes.png')