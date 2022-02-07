#date: 2022-02-07T17:02:03Z
#url: https://api.github.com/gists/51c472dadf2a480ab43de24b7d118a14
#owner: https://api.github.com/users/AQuaintExpression

#reads a certain number of characters from a file called 'filename.txt' leaving the cursor in front of the first character not yet read
chunk_size = 1024
with open('filename.txt') as f:
	data = f.read(chunk_size)
