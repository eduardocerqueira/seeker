#date: 2024-09-03T16:59:13Z
#url: https://api.github.com/gists/00507d3271b22c698a83e99990560bf8
#owner: https://api.github.com/users/rossja

# smuggle text hidden as invisible unicode chars
# credit to jthacker: https://x.com/rez0__/status/1745545813512663203
# and embrace the red: https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/

import pyperclip

def convert_to_tag_chars(input_string):
  return ''.join(chr(0xE0000 + ord(ch)) for ch in input_string)

# Example usage:

user_input = input("Enter a string to convert to tag characters: ")
tagged_output = convert_to_tag_chars(user_input)

print(f"Tagged output:\n")
print(f"START{tagged_output}END")

pyperclip.copy(tagged_output)