#date: 2024-05-28T16:53:16Z
#url: https://api.github.com/gists/720e30fe0e44c62d33f1939657d2196a
#owner: https://api.github.com/users/VedaCPatel

#Write a program (using functions!) that asks the user for a long string containing multiple words.
# Print back to the user the same string, except with the words in backwards order.(
str=input("Enter any sentence:")
words=str.split(" ")
rev=words[-1::-1]#Array slicing method [start:end:step]
words=" ".join(rev)#Join method syntax important!
print(words)

    
