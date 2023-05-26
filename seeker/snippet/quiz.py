#date: 2023-05-26T17:09:58Z
#url: https://api.github.com/gists/786465445514bc581fb196579dbb22cc
#owner: https://api.github.com/users/Eliit16

print("Welcome to my Computer quiz")
print("")
print("Do you want to play?")
print("Press 1 if yes")
print("Press 2 if no")

play = input()

if play == "1":
    name = input("Please Enter your name: ")
    print("")
    print("Lets proceed to the first question")
    score = 0
    print("")
elif play == "2":
    print("Goodbye, Have a nice day")
else:
    print("error")

print("1. What Does HTMl Mean?")
print("")
print("A. HyperTextMarkLanguage")
print("B HyparTextMarkupLanguage")
print("C. HyperTextMarkupLanguage")
print("D. HyperTextMarkapLanguage")

Answer1 = input("Please put your answer: ")

if Answer1 == "C":
    print("Correct")
    score +=1
else:
    print("Incorrect")

print("")

print("2. What does CSS stand for?")
print("")
print("A. Cascading StyleSheets")
print("B Cascading stayleSheets")
print("C. Cascading Styleshit")
print("D. Cascadoing Styleshits")

Answer2 = input("Please put your answer: ")

if Answer2 == "A":
    print("Correct!")
    score +=1
else:
    print("Incorrect")

print("")

print("3. What Does CPU Stand For?")
print("")
print("A. Central Process Unit")
print("B Central Process Units")
print("C. Cental Processing Unit")
print("D. Center Processing Unit")

Answer3 = input("Please put your answer: ")

if Answer3 == "C":
    print("Correct!")
    score +=1
else:
    print("Incorrect")

print("")

print("4. What Does RAM Stand for?")
print("")
print("A. Random Acess Memory")
print("B Random Access Memory")
print("C. Random Acess Memori")
print("D. Ramdom Acess Memories")

Answer4 = input("Please put your answer: ")

if Answer4 == "B":
    print("Correct")
    score += 1
else:
    print("Incorrect")

print("")

print("5. What Does PNG Stand for?")
print("")
print("A. Portable Networks Graphic")
print("B Portable Network Graphics")
print("C. Portables Network Graphic")
print("D. Protable Network Graphic")

Answer5 = input()

if Answer5 == "D":
    print("Correct")
    score += 1
else:
    print("Incorrect")




print("Congratulaion " + name + "," " Your Score is: " + str(score) + "/5")

