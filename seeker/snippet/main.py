#date: 2024-03-26T16:47:20Z
#url: https://api.github.com/gists/dc2e0a8786492e6e16ccf673599b2487
#owner: https://api.github.com/users/Aleppo21

def count_even_vowels(word):
    return sum(1 for i, letter in enumerate(word) if i % 2 == 0 and letter in "аеёиоуыэюя")

def main():
    sentence = input().lower()
    words = sentence.split()
    sorted_words = sorted(words, key=lambda word: (count_even_vowels(word), word))
    for word in sorted_words:
        print(word)

if __name__ == "__main__":
    main()