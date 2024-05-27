#date: 2024-05-27T17:06:50Z
#url: https://api.github.com/gists/27e240e7c828966169d82c0aacf317e7
#owner: https://api.github.com/users/VedaCPatel

#Write a program (function!) that takes a list and returns a new list that contains all the elements
# of the first list minus all the duplicates.
def list():
    l=[]
    n=int(input("How many numbers would you like to add to the list?:"))
    for i in range(1,n+1):
        a=int(input("Enter number"+str(i)+":"))
        b=l.append(a)
    print("The provided list is:",l)
    c=set(l) # Removing duplicates using the set() method!
    print("The revised list is:",c)
list()

