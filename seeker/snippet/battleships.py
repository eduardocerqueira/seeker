#date: 2022-04-27T16:58:46Z
#url: https://api.github.com/gists/c2e427e24653a679a22647d5742ac1d0
#owner: https://api.github.com/users/MerelCHT

if not (1 <= column <= 7) or not (1 <= row <= 7):
  print("Please enter a column and row between 1 and 7.")
elif (check_ship_exists(column, row)):
  print("A ship is already in that location! Try again.")
else:
  entry_valid = True