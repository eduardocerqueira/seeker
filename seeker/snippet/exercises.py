#date: 2024-11-19T17:07:01Z
#url: https://api.github.com/gists/9b99e4a21c927e54acaa5f2cdf1f3ee4
#owner: https://api.github.com/users/IceryDev

import math
import random
from time import sleep
import string

def is_int_or_float(number, text: str, mode: bool):
    if mode:
        while True:
            try:
                float(number)
            except ValueError:
                number = input(text)
                continue
            break
        return float(number) #Float
    else:
        while True:
            try:
                float(number)
            except ValueError:
                number = input(text)
                continue
            break
        return int(number) #Int

def p1():
    x = is_int_or_float(input("Enter an integer: "), "Invalid, enter a proper integer: ", False)
    if x > 0: print("Positive")
    elif x < 0: print("Negative")
    else: print("Zero")

def p2():
    x = is_int_or_float(input("Enter an integer: "), "Invalid, enter a proper integer: ", False)
    if x % 2 == 0: print("Even")
    else: print("Odd")

def p3():
    x = is_int_or_float(input("Enter your age: "), "Invalid, enter a proper age: ", False)
    if x >= 18: print("You are eligible to vote")
    else: print("You are not eligible to vote")

def p4():
    no_list: list[int] = []
    for a in range(3):
        no_list.append(is_int_or_float(input(f"Enter integer {a + 1}: "), "Invalid, enter a proper integer: ", False))
    print(max(no_list[0], no_list[1], no_list[2]))

def p5():
    x = is_int_or_float(input("Enter an integer: "), "Invalid, enter a proper integer: ", False)
    if x % 3 == 0 and x % 5 == 0: print(f"{x} is divisible by both 3 and 5.")
    elif x % 3 != 0 and x % 5 == 0: print(f"{x} is divisible by 5.")
    elif x % 3 == 0 and x % 5 != 0: print(f"{x} is divisible by 3.")
    else: print(f"{x} is not divisible by 5 and 3.")

def p6():
    x = is_int_or_float(input("Enter your marks: "), "Invalid, enter a proper mark: ", True)
    if x > 100: print("How did you get that mark????!!!!!")
    elif x >= 90: print("A")
    elif x >= 80: print("B")
    elif x >= 70: print("C")
    elif x >= 60: print("D")
    elif x < 60: print("F")

def p7():
    letter = input("Type a letter: ")
    if letter.lower() in ["a", "e", "u", "i", "o"]: print("That is a vowel.")
    elif letter.lower() in list(set(string.ascii_lowercase) - {"a", "e", "u", "i", "o"}): print("That is a consonant.")
    else: print("That is not an English letter!!")

def p8():
    x = is_int_or_float(input("Enter a year: "), "Invalid, enter a proper year: ", False)
    if (x % 4 == 0 and x % 100 != 0) or x % 400 == 0:
        print("Yay, that is a leap year!!")
    else:
        print("That is not a leap year.")

def p9():
    x = is_int_or_float(input("Enter a temperature in Celsius: "), "Invalid, enter a proper temperature: ", True)
    if x < -273: print("I see what you are doing... But I have thought of it. :)")
    else:
        a = (9 * x / 5) - 32
        print(f"The Fahrenheit value is {a} and it is ", end="")
        if a < 32: print("freezing.")
        elif a > 85: print("hot.")
        else: print("moderate.")

def p10():
    x = is_int_or_float(input("Enter an integer: "), "Invalid, enter a proper integer: ", False)
    print(f"The number is {int(math.log10(x)) + 1} digits.")

def p11():
    a = 1
    while a < 11:
        print(a)
        a += 1

def p12():
    a = 0
    s = 0
    while a < 100:
        a += 1
        s = s + a
    print(s)

def p13():
    s = 0
    while True:
        x = is_int_or_float(input("Enter an integer (0 to end process): "), "Invalid, enter a proper integer: ", False)
        s += x
        if x == 0: break
    print(s)

def p14():
    x = is_int_or_float(input("Enter an integer: "), "Invalid, enter a proper integer: ", False)
    b = 0
    for a in range(1, 11):
        b += 1
        if b % 3 == 0:
            print(f"{x} x {a} = {x * a}")
        else:
            print(f"{x} x {a} = {x * a}", end=" // ")

def p15():
    counter = 10
    while counter > 0:
        print(counter)
        sleep(1)
        counter -= 1
    print("Blast off!")

def p16():
    x = random.randint(1, 9)
    y = is_int_or_float(input("Enter an integer: "), "Invalid, enter a proper integer: ", False)
    while x != y:
        print("Wrong!")
        y = is_int_or_float(input("Enter an integer: "), "Invalid, enter a proper integer: ", False)
    print("Correct! Well done!")

while True:
    user_input = is_int_or_float(input("Which program do you want to run (1-16, 0 to exit)? "), "Enter an integer. ->", False)
    while not -1 < user_input < 17:
        user_input = is_int_or_float(input("Which program do you want to run (1-16)? "), "Enter an integer. ->", False)
    if user_input == 0: break
    else:
        globals()["p" + str(user_input)]()