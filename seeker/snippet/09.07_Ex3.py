#date: 2025-08-19T17:11:31Z
#url: https://api.github.com/gists/eaf720b69f7b37973c2716bf5288d623
#owner: https://api.github.com/users/Dmytro-Pin

#У списку цілих, заповненому випадковими числами, визначити мінімальний додатний і максимальний від'ємний елементи, 
#порахувати кількість від'ємних елементів, порахувати кількість додатних елементів, порахувати кількість нулів.
#Результати вивести на екран.

import random
num_list=[]
for i in range(5):
    x=random.randint(-100, 100)
    num_list.append(x)


max_neg=num_list[0]
for i in range(len(num_list)):
    temp=num_list[i]
    if temp>max_neg and temp<0:
        max_neg=num_list[i]
        temp=0
    temp=0
    continue

min_pos=num_list[0]
for i in range(len(num_list)):
    if num_list[i]<min_pos and temp>0:
        min_pos=num_list[i]
        temp=0
    temp=0
    continue

neg_count=0
for i in range(len(num_list)):
    if num_list[i]<0:
        neg_count+=1
    continue

pos_count=0
for i in range(len(num_list)):
    if num_list[i]>0:
        pos_count_count+=1
    continue

zero_count=0
for i in range(len(num_list)):
    if num_list[i]==0:
        zero_count+=1
    continue
