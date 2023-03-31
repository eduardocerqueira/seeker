#date: 2023-03-31T16:44:53Z
#url: https://api.github.com/gists/c44e1d315c887ad4241659b8247d297b
#owner: https://api.github.com/users/niranjan-exe

# Code for https://niranjan.blog/problem/kth-row-of-pascals-triangle

def pascal_triangle_row(k):
    # Create a list of length k+1 and initialize all its elements to 0, except for the first element, which is set to 1.
    row = [0] * (k+1)
    row[0] = 1
    
    # Iterate over all the rows up to and including the kth row.
    for i in range(1, k+1):
        # Compute each element of the row by adding the two elements above it in the previous row.
        # We start from the end of the row and work our way backwards, so that we don't overwrite values that we need in the next iteration of the loop.
        for j in range(i, 0, -1):
            row[j] += row[j-1]
    
    # Return the kth row.
    return row
