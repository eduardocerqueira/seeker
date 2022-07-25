#date: 2022-07-25T16:46:23Z
#url: https://api.github.com/gists/075c23e8d14d2ce45fd747f4b97386b9
#owner: https://api.github.com/users/KidZay

key = "lost"


def room1():
    print(
        "You awaken in a room with 3 doors in front of you the room has sparkles all over it making the room look like a picture of outer space.")
    print(
        "You can enter HINT to gain a possible assistance with the question but be aware not everything is here to help!")
    ans1 = input("Who won artist of the year in 2017? The Weekend, BTS, or Bruno Mars: ")
    if ans1 == "BTS":
        room2()
    elif ans1 == "The Weekend":
        room3()
    elif ans1 == "Bruno Mars":
        key == "Found"
        finalroom()
    elif ans1 == "HINT":
        print("We dont talk about_____")
        print("SZA song released in 2020 titled- The _______")


def room2():
    print("Wrong answer but you still have a chance.")
    print("The room is very up beat and energetic with LED lights flickering with all types of colors")
    print("There will be no hint here, everything you might need is in this room")
    ans2 = input("As of 2017 what is the most popular music genre?: House or R&B/Hip-Hop")
    if ans2 == "House":
        print("Incorrect you hear the sound of the door locking keeping you in forever")
    elif ans2 == "R&B/Hip-Hop":
        finalroom()


def room3():
    print("Wrong you still have a chance.")
    print("The room is covered in pictures in farm animals with most of them being horses")
    print("There will be no hints for this room everything you need is in here")
    rm3ques = input("Who won best new artist in 2020? Meg The Stallion, or Lizzo: ")
    if rm3ques == "Meg The Stallion":
        finalroom()
    elif rm3ques == "Lizzo":
        print(
            "Sadly No. You hear the sound of a lock keeping you from entering the next room. Your hint was with the room and there being more horses or Stallions.")


def finalroom():
    print(
        "Correct believe it or not this is the final room you will have to go through unless you get it wrong then it will be the last room you enter.")
    if key == "found":
        print("Oh....You...You found the key uhhh it definitely doesnt go to the door to leave...")
        print("*You slowly walk towards the door with the key in your hand*")
        print("Wait wait wait before you leave....")
        last = input("Do you like music Yes or No: ")
        if last == "No":
            print("hahaahahhahhahahhaha I tricked you ")
            print("You hear the door lock and try the key anyway but the door does not open. The End")
        elif last == "Yes":
            print("You walk up to the door and use the key to unlock it. The End")
    if key == "lost":
        print(
            "Oh good you didnt find the key im looking through the code and my creator apparently left the key behind the correct door in the first room for some reason. ")
        last = input("Alrighty last question do you like music?: ")
        if last == "Yes":
            print("The door opens allowing you to leave. The End")
        elif last == "No":
            print("You hear the sound of a door locking and a gate closing keeping you in forever. The End")


room1()