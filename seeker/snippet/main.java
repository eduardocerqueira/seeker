//date: 2023-03-31T16:51:56Z
//url: https://api.github.com/gists/64f33fadc1f541c88ae9194cbc8150cc
//owner: https://api.github.com/users/niranjan-exe

// Code for https://niranjan.blog/problem/kth-row-of-pascals-triangle

public List<Integer> getRow(int A) {
    // Create a List to store the row.
    List<Integer> row = new ArrayList<Integer>();
    
    // Set the first element to 1.
    row.add(1);
    
    // Iterate over all the rows up to and including the A-th row.
    for (int i = 1; i <= A; i++) {
        // Compute each element of the row by adding the two elements above it in the previous row.
        // We start from the end of the row and work our way backwards, so that we don't overwrite values that we need in the next iteration of the loop.
        for (int j = i; j >= 1; j--) {
            if (j == row.size()) {
                // If we're at the end of the row, add a new element.
                row.add(1);
            } else {
                // Otherwise, update the existing element by adding the two elements above it.
                row.set(j, row.get(j) + row.get(j-1));
            }
        }
    }
    
    // Return the A-th row.
    return row;
}
