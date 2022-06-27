#date: 2022-06-27T17:15:04Z
#url: https://api.github.com/gists/010867564729aa064893268fb142d633
#owner: https://api.github.com/users/elvanuzun

my_set1 = {"Asena", "Başak", "Bircan", "Emre"}
my_set2 = {"Emre", "Burak", "Bilal"}

my_set3 = my_set1 | my_set2

print(my_set3)

my_set4 = my_set1.union(my_set2)

print(my_set4)

'''
{'Bilal', 'Asena', 'Emre', 'Burak', 'Başak', 'Bircan'}
{'Bilal', 'Asena', 'Emre', 'Burak', 'Başak', 'Bircan'}
'''