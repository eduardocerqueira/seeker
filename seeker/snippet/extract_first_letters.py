#date: 2023-03-17T16:48:32Z
#url: https://api.github.com/gists/e2c2a6fe3fce77b744870545cd5a21a9
#owner: https://api.github.com/users/nasiegel88

import sys

# Accept sentence argument from command line
sentence = " ".join(sys.argv[1:])

# Split sentence into words
words = sentence.split()

# Extract first letter of each word and convert to lowercase
result = "".join(word[0].lower() for word in words)

print(result)  # Print resulting string

## Usage
#
#> python extract_first_letters.py "The quick brown fox jumps over the lazy dog"
#> tqbfjotld