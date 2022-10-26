#date: 2022-10-26T17:14:04Z
#url: https://api.github.com/gists/a6890fa6d6d84abcb22d2b66966a7c1f
#owner: https://api.github.com/users/kabirbg

#NOTE: you MUST to clone and build the master branch of skpy yourself!
#the moderation functions aren't implemented in the release version yet
#also, you MUST be an admin in the skype chat or else you will get a 403
from skpy import Skype
from getpass import getpass

username=input("Username: ")
password= "**********"
cont="yes"

while cont=="yes":
    a= "**********"
    key=input("Enter the chat name: ")
    chat=""
    x=0

    while x < 20 and chat=="": #run for loop up to 20 times
        for id in a.chats.recent():
            if id.startswith("19:") and a.chats[id].topic.find(key)!=-1:
                    print(id, a.chats[id].topic)
                    if input("Is this the correct chat? ") == "yes":
                        chat=a.chats[id]
        x+=1

    if not chat.moderated:
        if input("This chat is unmoderated. Make it moderated? ") == "yes":
            chat.setModerated()
            print(chat.topic, " is now a moderated Skype chat.")
    else:
        if input("This chat is moderated. Make it unmoderated? ") == "yes":
            chat.setModerated(False)
            print(chat.topic, " is now an unmoderated Skype chat.")

    cont=input("Continue with another chat? ") #allow looping without having to reenter credslooping without having to reenter creds