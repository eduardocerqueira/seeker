#date: 2024-11-14T17:12:23Z
#url: https://api.github.com/gists/f94456550c0ef4219fcec0436cdfc58d
#owner: https://api.github.com/users/dxdez

def is_isogram(string):
    # Convert the string to lowercase to ignore letter case
    string = string.lower()
    
    # Create a set to store unique letters
    unique_letters = set()
    
    # Iterate through each letter in the string
    for letter in string:
        # Check if the letter is already in the set
        if letter in unique_letters:
            return False
        else:
            # Add the letter to the set if not present
            unique_letters.add(letter)
    
    # If the loop completes without returning, it's an isogram
    return True