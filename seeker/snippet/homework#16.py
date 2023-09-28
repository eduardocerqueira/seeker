#date: 2023-09-28T16:55:50Z
#url: https://api.github.com/gists/8c1648293b1700b949c0126396a9e2bb
#owner: https://api.github.com/users/SashaKotton

print("Please enter color of the traffic light")
color = input()
if color == "red":
    print("Stop")
elif color == "yellow":
    print("Wait")
elif color == "green":
    print("Go")
else:
    print("Invalid color")