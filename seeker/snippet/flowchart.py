#date: 2023-10-24T16:44:54Z
#url: https://api.github.com/gists/071e64d9381f6bf7d304407e353fd6e3
#owner: https://api.github.com/users/DannyChaotix

is_home = "yes"

print("Ring ring!")

if is_home == "yes":

  print("Hello John, It's me.")
else:
  print("Hi John. Please call me back. I have something to ask you.")
  print("John calls you back. Ring ring!")

meal_response = input("Hi John, would you like to share a meal?")

if meal_response == "yes":
  print("Great, I'll see you at dinner tonight")
  
else:
  print("I see. You wouldn't like that.")
  beverage_response = input("Do you enjoy a hot beverage?")

  if beverage_response == "yes":

    print("Great, we can have a drink together tonight")

  else:

    print("I see. You wouldn't like that.")

    interest = input("Tell me one of your interests")

    while interest != "Physics" and interest != "Comic books":
      
      print("I see. I don't have that interest. I'm only interested in physics and comic books.")

      interest = input("Tell me one of your interests")
      

    if interest == "Physics" or interest == "Comic books":
      print("Let's work on something together!")

    else:
      print("Let's go to the comic book store sometime!")


print("The beginning of the friendship")