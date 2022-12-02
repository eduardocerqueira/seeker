#date: 2022-12-02T17:03:45Z
#url: https://api.github.com/gists/2b7a7d33aa44c9063093017a98b87557
#owner: https://api.github.com/users/grabovszky

data = open("input.txt").read()

score = 0
for line in data.split("\n"):
    player1, player2 = line.split()

    if player2 == "X":
        if player1 == "C":
            score += 6
        if player1 == "A":
            score += 3
        score += 1

    if player2 == "Y":
        if player1 == "A":
            score += 6
        if player1 == "B":
            score += 3
        score += 2

    if player2 == "Z":
        if player1 == "B":
            score += 6
        if player1 == "C":
            score += 3
        score += 3

print(score)
