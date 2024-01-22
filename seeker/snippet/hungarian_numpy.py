#date: 2024-01-22T16:56:14Z
#url: https://api.github.com/gists/ddb409c5e589a91afc6fcd852ea30356
#owner: https://api.github.com/users/secemp9

import numpy as np

def hungarian_algorithm(cost_matrix):
    N = len(cost_matrix)
    max_cost = np.max(cost_matrix)
    cost_matrix = max_cost - cost_matrix  # Convert to a profit matrix

    # Step 1: Subtract the row minimum from each row
    for i in range(N):
        cost_matrix[i] -= np.min(cost_matrix[i])

    # Step 2: Subtract the column minimum from each column
    for j in range(N):
        cost_matrix[:, j] -= np.min(cost_matrix[:, j])

    # Step 3: Cover all zeros with a minimum number of lines
    while True:
        covered_rows, covered_cols = cover_zeros(cost_matrix)
        if len(covered_rows) + len(covered_cols) == N:
            break

        # Step 4: Create more zeros
        min_uncovered_value = np.min(cost_matrix[~np.isin(range(N), covered_rows), :][:, ~np.isin(range(N), covered_cols)])
        cost_matrix[~np.isin(range(N), covered_rows), :] -= min_uncovered_value
        cost_matrix[:, covered_cols] += min_uncovered_value

    # Step 5: Find an optimal assignment
    assignment = np.zeros_like(cost_matrix, dtype=int)
    for _ in range(N):
        row, col = np.where((cost_matrix == 0) & (assignment == 0))
        for i, j in zip(row, col):
            if np.sum(assignment[i, :]) == 0 and np.sum(assignment[:, j]) == 0:
                assignment[i, j] = 1
                cost_matrix[i, :] = -1  # To avoid choosing the same row again
                cost_matrix[:, j] = -1  # To avoid choosing the same column again
                break

    return assignment * (max_cost - cost_matrix)

def cover_zeros(matrix):
    N = len(matrix)
    covered_rows = set()
    covered_cols = set()
    zeros = np.argwhere(matrix == 0)

    while zeros.size > 0:
        # Count zeros in rows and columns
        zero_count_row = np.array([np.sum(zeros[:, 0] == r) for r in range(N)])
        zero_count_col = np.array([np.sum(zeros[:, 1] == c) for c in range(N)])

        # Find row or column with the maximum number of uncovered zeros
        if np.max(zero_count_row) > np.max(zero_count_col):
            row = np.argmax(zero_count_row)
            covered_rows.add(row)
            zeros = zeros[zeros[:, 0] != row]
        else:
            col = np.argmax(zero_count_col)
            covered_cols.add(col)
            zeros = zeros[zeros[:, 1] != col]

    return covered_rows, covered_cols

# Example Usage
cost_matrix = np.array([[4, 2, 3], [2, 5, 1], [3, 4, 2]])
assignment = hungarian_algorithm(cost_matrix)

print("Original Cost Matrix:\n", cost_matrix)
print("Assignment Matrix:\n", assignment)