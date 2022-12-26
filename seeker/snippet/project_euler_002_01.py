#date: 2022-12-26T17:05:31Z
#url: https://api.github.com/gists/826d92e56daa3cb9607761f9c39ad7b7
#owner: https://api.github.com/users/mihiryerande

M = 4e6
S = 0

# Keep 2 previous Fibs at any given time
# Simply sum these to get the next Fib
fibs = [1, 0]
while fibs[1] <= M:
    # Add latest 3rd Fib to running sum
    S += fibs[1]

    # Step forward 3 elements in the sequence
    for _ in range(3):
        fibs.append(sum(fibs))
        fibs.pop(0)

print(S)