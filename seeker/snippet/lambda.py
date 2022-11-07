#date: 2022-11-07T17:16:49Z
#url: https://api.github.com/gists/e69de17d4ce2951f3a7baeb4d3b2c24f
#owner: https://api.github.com/users/Jesus-Vazquez-A

model_counts_serie = nissan.value_counts("model")

other_category = model_counts_serie[model_counts_serie < 10]

nissan.model = nissan.model.apply(lambda x: "Other" if x in other_category else x)