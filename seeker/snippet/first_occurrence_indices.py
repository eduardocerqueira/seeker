#date: 2025-11-11T16:58:44Z
#url: https://api.github.com/gists/09b1d9bcbecb4f859eb4a3b68ed24187
#owner: https://api.github.com/users/senarvi

def first_occurrence_indices(x: Tensor) -> Tensor:
    """Returns the index of the first occurrence of each unique value of the vector ``x``.

    Finds a sorted list of unique values in ``x``, and returns a list of indices to the first occurrence of each unique
    value. This is equivalent to ``np.unique(x, return_index=True)[1]``.

    Args:
        x: A list of values.

    Returns:
        A list of indices to the first occurrence of each unique value.

    """
    _, inverse_idxs, counts = torch.unique(x, sorted=True, return_inverse=True, return_counts=True)

    # inverse_idxs maps the items in x to unique values. Group items together that point to the same unique value,
    # starting from the smallest value. Use stable sort, so that the first item in each group is the first occurrence in
    # x.
    # For example:
    #   inverse_idxs = [0, 1, 4, 3, 2, 1, 0, 1, 2]
    #   grouped_idxs = [0, 6, 1, 5, 7, 4, 8, 3, 2]
    #   (2 indices for the first unique value, 3 indices for the second, 2 indices for the third, 1 index for the
    #   fourth, and 1 index for the fifth)
    grouped_idxs = torch.argsort(inverse_idxs, stable=True)

    # Now we just need to find where each group starts. First take the cumulative sum of the counts of the unique values.
    # For example:
    #   counts = [2, 3, 2, 1, 1]
    #   cumulative_sum = [2, 5, 7, 8, 9]
    cumulative_sum = counts.cumsum(0)

    # In a sorted list, the index of the first unique value is 0, and after that, the cumulative sum of the counts tells
    # the index of the next value.
    # For example:
    #   cumulative_sum = [2, 5, 7, 8, 9]
    #   new_value_idxs = [0, 2, 5, 7, 8]
    group_start_idxs = torch.cat((torch.tensor([0]), cumulative_sum[:-1]))

    # Now the first item in each group in grouped_idxs tells the index of the first occurrence of the value.
    return grouped_idxs[group_start_idxs]