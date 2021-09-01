#date: 2021-09-01T13:27:36Z
#url: https://api.github.com/gists/4cc12709f36b061219bc4790ea663389
#owner: https://api.github.com/users/orenmatar

n_mat = np.hstack([np.zeros((len(matrix), 1)), matrix])[:, :-1]  # elements moved by one, and zeros at the start
matrix = (n_mat + matrix) / 2