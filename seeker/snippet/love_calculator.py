#date: 2024-05-01T16:59:40Z
#url: https://api.github.com/gists/9c27c4702f4f6b633a2a97fd2ca2215f
#owner: https://api.github.com/users/Hardey1774

print("THIS IS A LOVE CALCULATOR")
user = input("YOU WANNA GIVE IT A TRY? (yes or no): ")

if user.lower() == "yes":
    print("Alright, Let's go")
    print("")
    name1 = input("Enter your name: ")
    name2 = input("Enter your spouse's name: ")

    combine_names = name1 + name2 
    cap_name = combine_names.upper()

    T = cap_name.count("T")
    R = cap_name.count("R")
    U = cap_name.count("U")
    E = cap_name.count("E")

    L = cap_name.count("L")
    O = cap_name.count("O")
    V = cap_name.count("V")
    E = cap_name.count("E")

    first_digit = T + R + U + E
    second_digit = L + O + V + E
    total_score = int(str(first_digit) + str(second_digit))



    if total_score < 10 or total_score > 90:
        print("Your score is ", total_score, "You are like coke and mentos")
    elif total_score > 40 and total_score <= 50:
        print("Your score is ", total_score,"\n","You are good together")
    else:
        print("Your score is ", total_score)

else:
    print("Goodbye.")



