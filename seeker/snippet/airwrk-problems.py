#date: 2023-07-11T17:05:48Z
#url: https://api.github.com/gists/0156ddc4e3f48156c17581ddbf491f23
#owner: https://api.github.com/users/echo-akash

#problem-1

#process breakdown

#divide the number by 1 - so we get the number of steps required in total if only 1 step is followed

#scenario for 1 step
n = 3
s1 = int(n/1)
# print(s1)
i = 0
while (i<s1):
  # print('1-step')
  i = i + 1

#divide the number by 2 - so we get the number of steps required in total if only 2 step is followed
#scenario for 2 steps

print_list = []

n = 9
s2 = int(n/2)
print(s2)
i = 0
while (i<s2):
  # print('2-step')
  print_list.append('2-step')
  i = i + 1

#the rest count of steps for 2 steps process is to subtract output times 2 from the number
s2s1 = n - (s2*2)
print(s2s1)
i = 0
while (i<s2s1):
  # print('1-step')
  print_list.append('1-step')
  i = i + 1

#print 2 step s2 times and 1 step s2s1 times
#we need to do combination of 2 step and 1 step and then print - how can we do that?
#insert 2 step into a list s2 times and 1 step into the same list s2s1 times
#pick any item from the list at random order and print the list

print(print_list)
# for i in print_list




#problem-2

#process breakdown 

# #divide the number by 10 - make it integer and get the first digit of the number
# n= 38
# d1 = int(n/10)
# print(d1)


# #multiply the first digit by 10 and subtract it from the main number
# d2 = n - (d1*10)
# print(d2)

# #sum of two digits
# n2 = d1+d2
# print(n2)

# #repeat the above process again
# n2_d1 = int(n2/10)
# print(n2_d1)

# n2_d2 = n2 - (n2_d1*10)
# print(n2_d2)

# n3 = n2_d1 + n2_d2
# print(n3)

# #divide n3 by 10
# n4 = n3/10
# print(n4)

# #if n4 is smaller than 1, then stop
# if (n4<1):
#   print('done')
# else:
#   print('repeat same steps again')


n = 38
while (int(n/10)) != 0:
  print(n)
  d1 = int(n/10)
  print(d1)
  d2 = n - (d1*10)
  print(d2)
  n = d1+d2
  print(n)

# int(n/10)




#problem-3

j = 'aA'
s = 'aAAbbbb'
# j = 'z'
# s = 'ZZ'

#convert chars of j into list and chars of s into list

j_list = list(j)
print(j_list)

s_list = list(s)
print(s_list)


#iterate items of j list and check how many times it is available in s list
count = 0
for j in j_list:
  print(j)
  for s in s_list:
    print(s)
    if (j == s):
      count = count + 1

print(count)