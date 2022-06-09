#date: 2022-06-09T17:16:19Z
#url: https://api.github.com/gists/1f4d45e92ede867685041058a0f8d2de
#owner: https://api.github.com/users/iswamik

""""
Let's explore how to choose a random item from a list.
"""
import random

# First, let's create a list of Avengers!
the_avengers = ["Iron Man", "Captain America", "Thor", "Hulk", "Black Widow"]

# Now, let's pick a random avenger from the list.
chosen_avenger = random.choice(the_avengers)

# Now, it's time to announce the chosen avenger to the world!
print(f"The chosen Avenger for the new mission is '{chosen_avenger}'")
