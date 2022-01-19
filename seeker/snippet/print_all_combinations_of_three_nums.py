#date: 2022-01-19T16:58:18Z
#url: https://api.github.com/gists/1a6e9d3bf7b3fe6190e046c15e9c2a9f
#owner: https://api.github.com/users/PythonRulz

'''

Problem Description
The program takes three distinct numbers and prints all possible combinations from the digits.

Problem Solution
1. Take in the first, second and third number and store it in separate variables.
2. Then append all the three numbers to the list.
3. Use three for loops and print the digits in the list if none of their indexes are equal to each other.
4. Exit.

'''
complete = False
while  not complete:    
    nums_list = []
    for i in range (3):
        nums_list.append(int(input(f"Enter number {i + 1} and no duplicates!!: ")))
    if nums_list[0] == nums_list[1] or nums_list[0] == nums_list[2] or nums_list[1] == nums_list[2]:
        print("I said 'No duplicates!!!'  TRY AGAIN")
    else:
        complete = True
        

for i in range (3):
    for x in range(3):
        for y in range(3):
            if (i != x and i != y and y != x):
                print(nums_list[i], nums_list[x], nums_list[y])
                      