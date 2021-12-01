#date: 2021-12-01T17:03:44Z
#url: https://api.github.com/gists/1643dca385c0c17d6d8c77de11eb2d12
#owner: https://api.github.com/users/JackWillz

def normalization(row):
    row = [np.log(x) for x in row]
    x_min, x_max = np.max(row), np.min(row)
    return [(x - x_min) / (x_max - x_min) for x in row]

list_of_stats = [normalization(list(top_champ_df[x])) for x in stats]
XT = np.array(list_of_stats)
X = np.transpose(XT)