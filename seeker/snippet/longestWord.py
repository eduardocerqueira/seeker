#date: 2024-02-01T16:54:37Z
#url: https://api.github.com/gists/beb21fc09fead52398c6856ecb088135
#owner: https://api.github.com/users/mighty-odewumi

def find_longest_word(filename):
    longest_word = ""
    with open(filename, "r") as file:
        for line in file:
            for word in line.split():
                word = word.strip(".,?!;:")
                if len(word) > len(longest_word):
                    longest_word = word
    return longest_word

# Test the function
filename = input("Enter the filename: ")
longest_word = find_longest_word(filename)
print("Longest word:", longest_word)
