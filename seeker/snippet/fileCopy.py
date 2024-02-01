#date: 2024-02-01T16:51:48Z
#url: https://api.github.com/gists/173061e7bdb705f217b704ed4d512816
#owner: https://api.github.com/users/mighty-odewumi

# Open file1 for reading
with open("file1.txt", "r") as file1:
    # Read the contents of file1
    file1_contents = file1.read()

# Open file2 for writing
with open("file2.txt", "w") as file2:
    # Write the contents of file1 into file2
    file2.write(file1_contents)

print("Contents of file1 copied into file2 successfully.")
