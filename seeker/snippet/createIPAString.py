#date: 2024-03-12T16:56:09Z
#url: https://api.github.com/gists/b0427b83db51d529d6b9d3aa099f8c4e
#owner: https://api.github.com/users/willwade

import eng_to_ipa as ipa

# The word you want to convert
word = "dog"

# Convert the word to its IPA representation
word_ipa = ipa.convert(word)

# Format the IPA representation for SSML
# Note: SSML tags are used here for illustrative purposes; actual usage depends on the SSML interpreter's capabilities
ssml_output = f"<phoneme alphabet='ipa' ph='{word_ipa}'>{word}</phoneme>"

print(ssml_output)
