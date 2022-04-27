#date: 2022-04-27T17:14:38Z
#url: https://api.github.com/gists/bab8293fd1d02676467e72d7622c491b
#owner: https://api.github.com/users/PushkraJ99

# Program to check if a string is palindrome or not

string = input("Input the String:")

# make it suitable for caseless comparison
string = string.casefold()

# reverse the string
rev_str = reversed(string)
# check if the string is equal to its reverse
if list(string) == list(rev_str):
   print("The string is a Palindrome.")
else:
   print("The string is not a Palindrome.")