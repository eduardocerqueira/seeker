#date: 2022-09-16T21:55:37Z
#url: https://api.github.com/gists/a9a1e4eff3f7fd5bd781524595adea5e
#owner: https://api.github.com/users/Abusagit

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    print("x_norm.shape:", x_norm.shape, "\n")
    
    # Divide x by its norm.
    x = x / x_norm

    return x