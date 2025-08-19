#date: 2025-08-19T17:11:31Z
#url: https://api.github.com/gists/eaf720b69f7b37973c2716bf5288d623
#owner: https://api.github.com/users/Dmytro-Pin

#Користувач із клавіатури вводить список цілих чисел. 
#Необхідно визначити максимальне і мінімальне значення у списку. 
#Результати вивести на екран.

num_list=[]
while True:
    user_inp=int(input('Введіть ціле число(для завершення введіть 0): '))
    if user_inp ==0:
        break
    num_list.append(user_inp)

max_n=num_list[0]
for i in range(len(num_list)):
    temp=num_list[i]
    if temp>max_n:
        max_n=num_list[i]
        temp=0
    temp=0
    continue

min_n=num_list[0]
for i in range(len(num_list)):
    if num_list[i]<min_n:
        min_n=num_list[i]
        temp=0
    temp=0
    continue

print('Максимальне число: ', max_n)
print('Мінімальне число: ', min_n)
