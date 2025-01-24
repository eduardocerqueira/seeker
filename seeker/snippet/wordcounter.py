#date: 2025-01-24T17:03:37Z
#url: https://api.github.com/gists/6ea702f80eb5b13d0dc57cbaee003b70
#owner: https://api.github.com/users/thomastschinkel

# word counter

text = input("Enter text to count the words: ").strip()
words = len(text.split())
if words == 1:
    print(f"The text: \"{text}\" has {words} word.")
else:
    print(f"The text: \"{text}\" has {words} words.")