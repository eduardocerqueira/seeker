#date: 2021-12-28T16:49:12Z
#url: https://api.github.com/gists/1c31fc751512d9097e26276ddd94e791
#owner: https://api.github.com/users/Jacalz


def roundrobin(seqs: list) -> list:
    """ roundrobin returns a list with items from the sublists
    added while alternating between all the sublists. """
    
    result = []
    index = 0    
    empty = 0

    # Loop while the empty sublists are less than all sublists.
    while empty < len(seqs):

        # We want to iterate over the same index in all sublists.
        for seq in seqs:
            if index < len(seq):
                result.append(seq[index])
            else:
                empty += 1 # One more sublist is empty.

        index += 1 # Look at the next index,

    return result

if __name__ == "__main__":
    assert roundrobin([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 4, 7, 2, 5, 8, 3, 6, 9]
    assert roundrobin([[1, 2, 3], [0], ["abc", 6]]) == [1, 0, "abc", 2, 6, 3]
    assert roundrobin([[]]) == []

