#date: 2023-12-08T17:08:06Z
#url: https://api.github.com/gists/081b08e6446b05e38b33e35e5dc09c40
#owner: https://api.github.com/users/sanjithpk

import re, math

with open("8.txt") as file:
    directions, data = file.read().split("\n\n")
    data = data.splitlines()

network = {}
dirIndex = dict(L=0, R=1)
l = len(directions)

for line in data:
    node, elements = line.split(" = ")
    network[node] = tuple(re.findall(r'(\w+)', elements))

def solve(cur):
    steps, part1 = 0, type(cur) is str 
    indices = [None] * len(cur)
    while True:
        direction = directions[steps % l]
        index = dirIndex[direction]
        if part1:
            cur = network[cur][index]
            steps += 1
            if cur == "ZZZ":
                break
        else:
            for i, node in enumerate(cur):
                cur[i] = network[node][index]
                if cur[i][2] == "Z":
                    indices[i] = steps + 1
            steps += 1
            if not any(i == None for i in indices): break
    return steps if part1 else math.lcm(*indices)

part1 = "AAA"
part2 = [node for node in network if node[2] == "A"]

print(solve(part1), solve(part2))