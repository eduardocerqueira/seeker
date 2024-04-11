#date: 2024-04-11T17:11:23Z
#url: https://api.github.com/gists/726532c6b03c896adc8fb307cfba4c5c
#owner: https://api.github.com/users/booshananamudara

def is_armstrong_number(number):
    str_number = str(number)
    num_digits = len(str_number)
    armstrong_sum = sum(int(digit) ** num_digits for digit in str_number)
    return armstrong_sum == number

def find_armstrong_numbers(start, end):
    armstrong_numbers = []
    for num in range(start, end + 1):
        if is_armstrong_number(num):
            armstrong_numbers.append(num)
    return armstrong_numbers

if _name_ == "_main_":
    start_range = 0
    end_range = 100000
    armstrong_numbers = find_armstrong_numbers(start_range, end_range)
    print("Armstrong numbers between", start_range, "and", end_range, "are:")
    print(armstrong_numbers)