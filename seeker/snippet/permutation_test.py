#date: 2022-07-15T17:05:10Z
#url: https://api.github.com/gists/b8d570f720e984448a3cb331cf950673
#owner: https://api.github.com/users/JAugusto97

def permutation_test(A, B, num_iterations):
    mean_diff = np.mean(B) - np.mean(A)
    count_bigger = 0
    for _ in range(num_iterations):
        samples = np.concatenate((A,B))
        idx_samples = [idx for idx in range(len(samples))]

        A_idxs = np.random.choice(idx_samples, size=len(A), replace=False)
        B_idxs = np.array(list(set(idx_samples).difference(set(A_idxs))))

        A_samples, B_samples = samples[A_idxs], samples[B_idxs]
        sample_mean_diff = np.mean(B_samples) - np.mean(A_samples)
        if sample_mean_diff >= mean_diff:
            count_bigger += 1

    p = count_bigger/num_iterations
    return p