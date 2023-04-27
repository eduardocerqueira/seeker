#date: 2023-04-27T16:50:26Z
#url: https://api.github.com/gists/b35de3dd22589b89fb12528a966bbbfc
#owner: https://api.github.com/users/SarpantKeltiek

import os
import sys
import tokenize

#verif if word is in camelCase
def is_camel_case(word):
  if (word[0].isupper()):
    return False

  for letter in word:
    if (letter.isupper()):
      return True

  return False

#convert a word to snake_case
def convert_to_snake_case(word):

  cnt = 0
  for letter in word:

    if letter in "()":
      return word

    if letter.isupper():

      new_case = '_' + letter.lower()
      word = word[:cnt] + new_case + word[cnt+1:]
      cnt +=1
    cnt += 1

  return word      
 
def __main__():

  if (len(sys.argv) < 2):
    print("Error: input file required")
    sys.exit(1)

  file_name = sys.argv[1]
       
  f = open(file_name, 'rb')
  ff = open(file_name, 'r')

  tokens = "**********"
  filedata = ff.read()
  counter = 0

 "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"[ "**********"0 "**********"] "**********"  "**********"= "**********"= "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********". "**********"N "**********"A "**********"M "**********"E "**********") "**********": "**********"
      word = "**********"
      if (is_camel_case(word)):
        counter += 1
        new_word = convert_to_snake_case(word)
        filedata = filedata.replace(word, new_word)

  if (counter):
    with open('new_test.py', 'w') as file:
      file.write(filedata)
    print(counter," camelCase were eliminated !")

  f.close()
  ff.close()

if __name__ == "__main__":
  __main__()