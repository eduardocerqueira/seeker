#date: 2022-01-26T17:09:29Z
#url: https://api.github.com/gists/b0a745cdfba63b3338998cfce8c75b14
#owner: https://api.github.com/users/PythonRulz

'''

Problem Description
The program takes a number and checks if it is a strong number.

Problem Solution
1. Take in an integer and store it in a variable.
2. Using two while loops, find the factorial of each of the digits in the number.
3. Then sum up all the factorials of the digits.
4. Check if the sum of the factorials of the digits is equal to the number.
5. Print the final result.
6. Exit.

Example:

Input  : n = 145
Output : Yes
Sum of digit factorials = 1! + 4! + 5!
                        = 1 + 24 + 120
                        = 145

Input :  n = 534
Output : No

'''

def is_strong_num(num):
    strong_nums = []
    for x in range (1, num+1):
        total = 0
        temp = x
        while x:
            remainder = x%10
            fac = 1
            i = 1
            while(i <= remainder):
                fac *= i
                i += 1
            total += fac
            x = x //10
            
        if total == temp:
            strong_nums.append(temp)
             
    if strong_nums:
        return (f"Here is the list of strong numbers: {strong_nums}")
    else:
        return (f"There are no strong numbers in that range")
        
        

def main():
    num =  int(input("Enter a number to determine the strong numbers in the range: "))
    result = is_strong_num(num)
    print(result)
    
    

main()