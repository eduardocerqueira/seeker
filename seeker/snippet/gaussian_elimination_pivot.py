#date: 2022-01-12T17:18:32Z
#url: https://api.github.com/gists/269be1c619f2350195f3b651368d3892
#owner: https://api.github.com/users/ekralc

def gaussian_elimination(A, b):
    """
    Converts a normal linear system of equations to upper triangular form

    A is a non-singular n x n matrix containing coefficients
    b is an n-vector containing the values
    """

    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    for i in range(n-1):
        # Swap rows to avoid zero division
        swap = i
        largest_mag = np.abs(A[swap][i])

        # Find the row with the largest magnitude
        for x in range(i, n):
            mag = np.abs(A[x][i])
            if (mag > largest_mag):
                largest_mag = mag
                swap = x

        # Row swapping
        if i != swap:
            A[[i, swap]] = A[[swap, i]]
            b[i], b[swap] = b[swap], b[i].copy()

        for j in range(i + 1, n):
            multiplier = A[j][i] / A[i][i]

            A[j] -= multiplier * A[i]
            b[j] -= multiplier * b[i]

            # Explicitly set this to 0 to avoid rounding errors
            A[j, i] = 0.0

    return A, b