#date: 2023-05-05T16:49:55Z
#url: https://api.github.com/gists/2b57267f8543089d306d7ea7127df42b
#owner: https://api.github.com/users/saeiddrv

# Python in Persian: https://python.coderz.ir
# Code challenges, Basic
# This Python program accepts an integer as input from the user to draw Khayyam Pascal's triangle.


# Accept the number of rows
n = int(input("Enter the number of rows: "))

# Initialize the triangle with the first row
triangle = [[1]]

# Add the remaining rows to the triangle
for i in range(1, n):
    row = [1]
    for j in range(1, i):
        row.append(triangle[i-1][j-1] + triangle[i-1][j])
    row.append(1)
    triangle.append(row)

# Print the triangle
for row in triangle:
    print(" " * (n-len(row)), end="")
    for num in row:
        print(f"{num} ", end="")
    print()
