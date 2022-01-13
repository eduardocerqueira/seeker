#date: 2022-01-13T17:02:06Z
#url: https://api.github.com/gists/8a0a93350853e0490bc9b29b41a7659d
#owner: https://api.github.com/users/Kodzila1

number = int(input("Please enter the number: "))
list= input("even or odd?")
if list=="even":
    for i in range(0, number+1, 2):
        print(i)
elif list=="odd":
    for j in range(1, number+1,2):
        print(j)
#elif list=="stop":
        #continue
else:
    print("Incorrect input")

