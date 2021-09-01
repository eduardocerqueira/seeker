#date: 2021-09-01T13:04:44Z
#url: https://api.github.com/gists/746b6ce7606e4b96233051f2250fc026
#owner: https://api.github.com/users/eveem

t = int(input())

for r in range(1, t + 1):
    n = int(input())
    m = []
    for _ in range(n):
        m.append(input())

    fin_by = set()

    for i in range(n):
        p_col = 0
        x_col = 0

        p_row = 0
        x_row = 0

        s_col = []
        s_row = []

        for j in range(n):
            if m[j][i] == ".":
                p_col += 1
                s_col.append((j, i))
            if m[j][i] == "X":
                x_col += 1
            if p_col + x_col == n:
                fin_by.add(tuple(s_col))

            if m[i][j] == ".":
                p_row += 1
                s_row.append((i, j))
            if m[i][j] == "X":
                x_row += 1
            if p_row + x_row == n:
                fin_by.add(tuple(s_row))

    result = {i: 0 for i in range(1, n + 1)}

    for i in fin_by:
        result[len(i)] += 1

    ans = 0
    mini = float("inf")

    for i in range(1, n + 1):
        if i in result and result[i] > 0:
            ans = i
            mini = result[i]
            break
    if ans == 0 and mini == float("inf"):
        print(f"Case #{r}: Impossible")
    else:
        print(f"Case #{r}: {ans} {mini}")
