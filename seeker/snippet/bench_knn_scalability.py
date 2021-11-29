#date: 2021-11-29T17:05:00Z
#url: https://api.github.com/gists/8e8e7e20c418aabb9e763aa3a7a34091
#owner: https://api.github.com/users/ogrisel

from pathlib import Path
import sys
from time import perf_counter
from threadpoolctl import threadpool_limits
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from joblib import Memory
import pandas as pd
import matplotlib.pyplot as plt

run_label = "gh_21462"
use_n_jobs = False
n_samples_train = int(1e6)
n_samples_test = int(1e4)
n_features = 100
n_neighbors = 10

filename = f"bench_knn_scalability_{run_label}.json"
if "--plot-results" in sys.argv:
    df = pd.read_json(filename)
    fig, ax = plt.subplots()
    ax.loglog(df["n_workers"], df["n_workers"], linestyle="--", color="black", label="linear", alpha=.5)
    ax.loglog(df["n_workers"], df["speedup"], label=run_label)
    ax.set(
        xlabel="# workers",
        ylabel="speed-up",
        xticks=df["n_workers"],
        xticklabels=df["n_workers"],
        yticks=df["n_workers"],
        yticklabels=[str(i) + "x" for i in df["n_workers"]],
        title="Scalability of k-NN"
    )
    plt.legend()
    plt.show()
    sys.exit(0)

m = Memory(location=".")
make_blobs = m.cache(make_blobs)


X, y = make_blobs(
    n_samples=n_samples_train + n_samples_test, n_features=n_features, random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=n_samples_test, random_state=0
)


ref_time = None
records = []
for n_workers in [1, 2, 4, 8, 16, 32, 64]:
    if use_n_jobs:
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_workers).fit(X_train)
        tic = perf_counter()
        nn.kneighbors(X_test)
        delta = perf_counter() - tic
    else:
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_workers).fit(X_train)
        with threadpool_limits(limits=n_workers):
            tic = perf_counter()
            nn.kneighbors(X_test)
            delta = perf_counter() - tic
    if ref_time is None:
        ref_time = delta
    speedup = ref_time / delta

    print(f"n_workers={n_workers}: duration={delta:.3f}s, speed-up: {speedup:.1f}x")
    records.append({
        "n_workers": n_workers,
        "duration": delta,
        "speedup": speedup,
    })
records = pd.DataFrame(records)
records.to_json(filename)