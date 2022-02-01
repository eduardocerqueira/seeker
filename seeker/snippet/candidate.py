#date: 2022-02-01T16:57:48Z
#url: https://api.github.com/gists/ca68efd9510ee56f6b7d62cce5eca1cf
#owner: https://api.github.com/users/suhanacharya

# from pandas import DataFrame
import pandas as pd

# get data from EnjoySports.csv
data = pd.read_csv("EnjoySports.csv")

# get the list of attributes
concepts = data.values[:, :-1]
target = data.values[:, -1]

# learn function for candidate elimination
def learn(concepts, target):
    specific_h = concepts[0].copy()

    len_specific = len(specific_h)
    print(len_specific)
    # general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    general_h = [["?" for i in range(len_specific)] for i in range(len_specific)]
    # print(general_h)
    # general_h = [["?"] * len_specific] * len_specific
    print(general_h)
    # print(specific_h)

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = "?"
                    general_h[x][x] = "?"
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = "?"

    indices = [
        i
        for i, val in enumerate(general_h)
        if val == ["?" for i in range(len(specific_h))]
    ]
    for i in indices:
        general_h.remove(["?" for i in range(len(specific_h))])
    return specific_h, general_h


s_final, g_final = learn(concepts, target)
print("Final S:", s_final)
print("Final G:", g_final)
