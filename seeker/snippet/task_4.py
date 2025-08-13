#date: 2025-08-13T16:40:41Z
#url: https://api.github.com/gists/c16e305a437ee7787571bd8dbb39ee8c
#owner: https://api.github.com/users/DanielIvanov19

def lzip(*iterables):
    # Turning every input iterable into list for easy indexing
    data = [list(it) for it in iterables]
    max_len = max(len(it) for it in data)

    for i in range(max_len):
        # For every position get the element from each iterable which is repeated when needed
        yield tuple(seq[i % len(seq)] for seq in data)