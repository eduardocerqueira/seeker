#date: 2024-02-13T16:56:38Z
#url: https://api.github.com/gists/74f9f0d53cb64ee3f6a6f4e8d10e2724
#owner: https://api.github.com/users/Archonic944

# Type a number and hit enter for that number's look-and-say product.
# Press enter without entering a number to get the next step after the previous product.
def look_and_say(num):
    prev_val = None
    occurrence = 0
    new_seq = ""
    for i in map(int, str(num)):
        if i == prev_val:
            occurrence += 1
        elif prev_val is None:
            prev_val = i
            occurrence = 1
        else:
            new_seq += str(occurrence) + str(prev_val)
            prev_val = i
            occurrence = 1
    new_seq += str(occurrence) + str(prev_val) #for last item
    return int(new_seq)

uin = ""
prev_las = 1
while uin != "stop":
    uin = input("")
    if uin == "":
        prev_las = look_and_say(prev_las)
    else:
        prev_las = look_and_say(uin)
    print(prev_las)