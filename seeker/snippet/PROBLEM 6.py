#date: 2023-07-20T17:04:21Z
#url: https://api.github.com/gists/43ba0c3c6a2ce6c94c1c6e1e55fcfcec
#owner: https://api.github.com/users/semihkalkandelen

"""Take the two perpendicular sides (a, b) of a right triangle from the user
and try to find the length of the hypotenuse."""
AB = float(input("Please insert value of [AB]"))
BC = float(input("please insert value of [BC]"))
AC = ((BC**2) + (AB**2))**(1/2)
print("hypotenuse of ABC triangle is:" , AC)