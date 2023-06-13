#date: 2023-06-13T16:52:05Z
#url: https://api.github.com/gists/a972d2eee996c66f36f0dcb6a9ef8b07
#owner: https://api.github.com/users/HaileyStorm

import numpy as np

# Load the binary file and reshape it to 256x256
with open('data17square.bin', 'rb') as f:
    data = np.fromfile(f, dtype=np.uint8)
data = np.unpackbits(data).reshape(256, 256)

# Define the magic square
magic_square = np.array([[7,8,5,6],[15,16,13,14],[1,2,3,4],[9,10,11,12]])

# Initialize an empty list to hold the sums
sums = []

# Loop over the data in 4x4 blocks
for i in range(0, 256, 4):
    for j in range(0, 256, 4):
        block = data[i:i+4, j:j+4]
        # Use the block as a mask on the magic square
        sum_ = np.sum(block * magic_square)
        # Then sum the selected magic square numbers
        sums.append(sum_)

# Convert the sums to binary and write to a new file
with open('data17square_magic_sums.bin', 'wb') as f:
    for sum_ in sums:
        hex_ = np.array([sum_], dtype=np.uint8).tobytes()
        f.write(hex_)
