#date: 2021-12-09T16:56:40Z
#url: https://api.github.com/gists/e8fd1d1d9010bcc1b1b5f7f09dda9378
#owner: https://api.github.com/users/alexppppp

print("\nArray b multiplied by boolean mask:")
print(b * b_mask_boolean)

print("\nArray b multiplied by inversed boolean mask:")
print(b * ~b_mask_boolean)

print("\nPart of array a (4th and 5th rows; 3rd, 4th, 5th and 6th columns) multiplied by boolean mask:")
print(a_copy[3:5,2:6] * b_mask_boolean)

print("\nPart of array a (4th and 5th rows; 3rd, 4th, 5th and 6th columns) multiplied by inversed boolean mask:")
print(a_copy[3:5,2:6] * ~b_mask_boolean)