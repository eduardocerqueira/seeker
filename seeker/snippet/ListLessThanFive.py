#date: 2022-01-10T16:54:52Z
#url: https://api.github.com/gists/c71d6e4e87ec0836f250d7f8506e23f3
#owner: https://api.github.com/users/rimuru72

#Main task is to print list a elements that are less than 5
#Task 0 is to make a new list that contains such elements then print the list
#Task 1 is to write the output of Main Task in one line
#Task 2 is to do the same as main task but all elements should be less than the user input number


a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

#Start of Extra Task 0
b = []
#Hold of Extra Task 0

for el in a:
    if el < 5:
        print(el)
        #Continue of Extra Task 0
        b.append(el)
print(b)
#End of Extra Task 0        

for el in a:
    if el < 5:
        #Start of Extra Task 1
        print(el, " ", end = '')
        #End of Extra Task 1
print()

#Start of Extra Task 2
user = int(input("Please enter a number: "))
for el in a:
    if el < user:
        print(el)
#End of Extra Task 2