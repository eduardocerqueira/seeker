#date: 2024-12-30T16:42:54Z
#url: https://api.github.com/gists/123f6bd00a190d2360049e2883b51a2b
#owner: https://api.github.com/users/blueWhale1202

min_sums = []

for i in pairings_sum:
    s = 0
    for j in range(len(i)):
        print('i:',i)
        print(dijktra(graph, i[j][0], i[j][1]))
        s += dijktra(graph, i[j][0], i[j][1])
    min_sums.append(s)


print(min_sums)