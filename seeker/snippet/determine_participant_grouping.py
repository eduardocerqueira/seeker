#date: 2023-05-23T16:39:24Z
#url: https://api.github.com/gists/dde0badedc8359921a53ea3bb58b5df7
#owner: https://api.github.com/users/MBrede

def determine_participant_grouping(i, sizes=[2]):
    """
    Determines the conditions a participant is in based on the given integer ID and group sizes.

    Args:
        i (int): The integer ID of the participant.
        sizes (list): A list of group sizes. Default is [2].

    Returns:
        list: A list of conditions that the participant belongs to.

    Explanation:
    This function calculates the conditions that a participant is in based on their integer ID and the provided group sizes.
    The participant's ID is divided into segments based on the cumulative product of the group sizes. Each segment represents a different level of grouping.

    For example, if sizes=[2, 3], the participant's ID will be divided into two segments. The first segment represents the outer group with size 2, and the second segment represents the inner group with size 3.
    The function uses modulo arithmetic and integer division to determine the condition index for each segment.

    The return value is a list of condition indices that correspond to the participant's position in each group.

    Note:
    - The group sizes must be positive integers.
    - The participant's ID should be a non-negative integer.
    - The function assumes that the group sizes are given in decreasing order of importance, where the first group will be the one changing the most often.
    """

    cumprod_sizes = [1]
    for size in sizes:
        cumprod_sizes.append(cumprod_sizes[-1] * size)
    
    return [i % s // cumprod_sizes[j-1] for j, s in enumerate(cumprod_sizes) if j != 0]
