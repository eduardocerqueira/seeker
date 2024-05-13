#date: 2024-05-13T17:08:58Z
#url: https://api.github.com/gists/ae307b42db95189bd55c0799a7a09942
#owner: https://api.github.com/users/CelilErenKalkan

alphabet = "abcdefghijklmnopqrstuvwxyz"

morse_code = {
    'a' : '.-', 'b' : '-...', 'c' : '-.-.', 'd' : '-..',
    'e' : '.', 'f' : '..-.', 'g' : '--.', 'h' : '....',
    'i' : '..', 'j' : '.---', 'k' : '-.-', 'l' : '.-..',
    'm' : '--', 'n' : '-.', 'o' : '---', 'p' : '.--.',
    'q' : '--.-', 'r' : '.-.', 's' : '...', 't' : '-',
    'u' : '..-', 'v' : '...-', 'w' : '.--', 'x' : '-..-',
    'y' : '-.--', 'z' : '--..'
}

# Get the string to translate

# Convert the message to lowercase.

# Iterate over the letters of the message.

  # If the character is a space, add '/' to the encoded message.


  # Otherwise, if the character is a letter, replace it with
  # its morse code representation.

def morse_translation(message):
  message = message.lower()
  words = message.split(" ")
  print(words)
  translation = ""

  for i in range(len(words)):
    for char in words[i]:
      if char in alphabet:
        translation += morse_code[char]
    if i < len(words) - 1:
      translation += "/"
  print(translation)

message = str(input("Please Enter Your Message: "))
morse_translation(message)