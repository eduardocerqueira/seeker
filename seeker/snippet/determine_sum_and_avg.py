#date: 2022-01-18T17:08:46Z
#url: https://api.github.com/gists/378db814cd8989318e14e2464d5b7bd0
#owner: https://api.github.com/users/PythonRulz

'''

Problem Description
The program takes the elements of the list one by one and displays the average of the elements of the list.

Problem Solution
1. Take the number of elements to be stored in the list as input.
2. Use a for loop to input elements into the list.
3. Calculate the total sum of elements in the list.
4. Divide the sum by total number of elements in the list.
5. Exit.

'''

# no error checking in this program

def calculate_avg (num_list):
    return sum(num_list) / len(num_list)

def create_list(length):
    nums_list = []
    for x in range (length):
        nums_list.append(int(input(f"Enter number {x + 1}  to add to the list: ")))
    return nums_list

def main():
    print("This program will ask the user to create a list of variable size\n"
          "and then determine the sum of the list and return the average")
    print()
    nums_list = create_list(int(input("How large do you want the list to be?: ")))
    print(f"The average of {nums_list} is {(calculate_avg(nums_list))}")

main()    
    