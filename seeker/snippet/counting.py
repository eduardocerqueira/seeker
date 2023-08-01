#date: 2023-08-01T16:35:42Z
#url: https://api.github.com/gists/d301ae2b0ba3eb82ef7dddae562e7284
#owner: https://api.github.com/users/matiasrasmussen

import pandas as pd

data = pd.DataFrame([
    ("pizz","pizza", 32),
    ("pizza", "pizza", 83),
    ("pi", "pizza", 12),
    ("piz", "pizza", 15),
    ("burger", "pizza", 2),
    ("burger", "burger", 120),
    ("burgr", "burger", 50),
    ("bur", "burger", 10),
], columns=["query", "category", "count"])


def P_query_given_cat(query: str, category: str) -> float:
    cat_counts = data.groupby("category").sum().loc[:, ["count"]]
    P_cat_given_query = (data.loc[(data["query"] == query) & (data["category"] == category), "count"].sum()) / (cat_counts.loc[category, "count"])
    return P_cat_given_query


if __name__ == '__main__':
    for row in data.iterrows():
        query = row[1]["query"]
        category = row[1]["category"]
        print("P( query = {} | category = {} ) = {} ".format(query, category, P_query_given_cat(query, category)))