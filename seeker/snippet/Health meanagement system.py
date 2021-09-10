#date: 2021-09-10T16:55:49Z
#url: https://api.github.com/gists/461f8298d9c263358c46f68d4f1b6fb5
#owner: https://api.github.com/users/junedsheikh

client_list = {1: "Bushra", 2: "Kaynat", 3: "Khubaid"}
lock_list = {1: "Exercise", 2: "Diet"}

def getdate():
    import datetime
    return datetime.datetime.now()

try:
    print("Select Client Name:")
    for key, value in client_list.items():
        print("Press", key, "for", value, "\n", end="")
    client_name = int(input())
    print("Selected Client : ", client_list[client_name], "\n", end="")
    op = int(input("Press 1 : Log\nPress 2 : Retrieve\n"))

    if op == 1:
        for key, value in lock_list.items():
            print("Press", key, ": log", value, "\n", end="")
        lock_name = int(input())
        print("Selected : ", lock_list[lock_name])
        f = open(client_list[client_name] + "_" + lock_list[lock_name] + ".txt", "a")
        k = 'y'
        while k != "n":
            print("Enter", lock_list[lock_name], "\n", end="")
            mytext = input()
            f.write("[ "+str(getdate())+"] : " + mytext + "\n")
            k = input("ADD MORE? y/n:")
            continue
        f.close()
    elif op == 2:
        for key, value in lock_list.items():
            print("Press", key, ": reteieve", value, "\n", end="")
        lock_name = int(input())
        print(client_list[client_name], "_", lock_list[lock_name], "Report : ", "\n", end="")
        f = open(client_list[client_name] + "_" + lock_list[lock_name] + ".txt", "rt")
        print(f.read())
        f.close()
    else:
        print("Invalid Input !")
except Exception as e:
    print("Wrong Input !")