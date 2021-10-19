#date: 2021-10-19T16:56:21Z
#url: https://api.github.com/gists/1914fb91f518c9a94cf6beb928a4510d
#owner: https://api.github.com/users/mesejo

from collections import defaultdict
from random import choices

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def weighted_sample_without_replacement(population, weights, k=1):
    #    https://stackoverflow.com/a/43649323/4001592
    weights = list(weights)
    positions = range(len(population))
    indices = []
    while True:
        needed = k - len(indices)
        if not needed:
            break
        for i in choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0.0
                indices.append(i)
    return [population[i] for i in indices]


data = [
    ("object_5", 0.99),
    ("object_2", 0.75),
    ("object_1", 0.50),
    ("object_3", 0.25),
    ("object_4", 0.01),
]

_, weights = zip(*data)
counts = defaultdict(lambda: defaultdict(int))
for _ in range(1000):
    sample = weighted_sample_without_replacement(data, weights, k=len(data))
    for i, (key, _) in enumerate(sample):
        counts[i][key] += 1

df = pd.DataFrame([[key, *value] for key, values in counts.items() for value in values.items()],
                  columns=["position", "label", "Counts"])

g = sns.catplot(
    data=df, kind="bar",
    x="position", y="Counts", hue="label",
    ci="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "Counts")
g.legend.set_title("")

plt.show()