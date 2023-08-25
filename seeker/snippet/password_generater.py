#date: 2023-08-24T16:36:31Z
#url: https://api.github.com/gists/b999df18e144885f1c486aa3749ddc8b
#owner: https://api.github.com/users/Harshil-230

#A SIMPLE PASSWORD GENERATOR
"""import necessary modules"""
from string import ascii_letters,digits
from random import choice

"""password generating function"""
def pswd(length=12,special_characters=True,numbers=True,alphabets=True):
    pswd=''
    all_chars=''
    if special_characters:
        all_chars+="!@#$%^&*()~`-=_+{}[]:;'<>,.?/"+'"'
    if numbers:
        all_chars+=digits
    if alphabets:
        all_chars+=ascii_letters
""" generating password of given length"""
    for i in range(length):
        pswd+=choice(all_chars)    
    return pswd
"""getting the required information from the user"""
length=input("Length of password: "**********"
n= "**********"
a= "**********"
s= "**********"
if n=='y':
    n=True
else:
    n=False
if a=='y':
    a=True
else:
    a=False
if s=='y':
    s=True
else:
    s=False
"""printing the generated password"""
print("Your generated password is: "**********"
else:
    a=False
if s=='y':
    s=True
else:
    s=False
"""printing the generated password"""
print("Your generated password is: "**********"h,s,n,a))