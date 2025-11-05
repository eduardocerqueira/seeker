#date: 2025-11-05T17:05:08Z
#url: https://api.github.com/gists/a256604b4aeb66d019d2ddb81e5bd2ac
#owner: https://api.github.com/users/Joycarey

score =int(input("Enter the score:"))
if score>=90:
    print("The Grade is A")
elif (score <=89 and score>=80):
    print("The Grade is B")
elif(score <=79 and score >=70):
    print("The Grade is C")
elif( score <=69 and score >=60):
    print("The Grade is D")
else:
    print("The Grade is F")
    
    # Displaying the first 10 natural numbers
print("The first 10 natural numbers are:")
for i in range(1, 11):  # Loop from 1 to 10
    print(i)
    
    digit=int(input("Enter the Number:"))
string=str(digit) #since integer doesn't have len()
count=len(string)
print("The number of digits in ",digit,"is :",count)