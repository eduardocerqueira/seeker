#date: 2023-11-29T16:45:39Z
#url: https://api.github.com/gists/e4a0282080bbf7d47db2f7b310fd3c74
#owner: https://api.github.com/users/ak1190506

sub1 = int(input("Math: "))
sub2 = int(input("Physics: "))
sub3 = int(input("Chemistry: "))
sub4 = int(input("History: "))
sub5 = int(input("English: "))
print()
total = sub1+sub2+sub3+sub4+sub5
print("Total score: ", total)
percentage = (total/500)*100
print("Percentage = ", percentage, "%")
