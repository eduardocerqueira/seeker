#date: 2022-02-14T17:01:33Z
#url: https://api.github.com/gists/9c344ac284afdc479723c9aa0aa21266
#owner: https://api.github.com/users/PythonRulz

'''

Problem Description
The program takes a dictionary and removes a given key from the dictionary.

Problem Solution
1. Declare and initialize a dictionary to have some key-value pairs.
2. Take a key from the user and store it in a variable.
3. Using an if statement and the in operator, check if the key is present in the dictionary.
4. If it is present, delete the key-value pair.
5. If it isn’t present, print that the key isn’t found and exit the program.
6. Exit.

'''       
        
    
def main():
    my_dict = {'Num1':10, 'Num2':20, 'Num3':100, 'Num4':150, 'Num5':1}
    key = input("Enter the key to be removed: ").capitalize()
    if key in my_dict:
        del my_dict[key]
        print(f"The dictionary without the key {key} is {my_dict}")
    else:
        print(f"The key {key} is not present in the dictionary {my_dict}")           
    
    
main()
