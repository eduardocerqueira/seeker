#date: 2025-08-19T17:11:31Z
#url: https://api.github.com/gists/eaf720b69f7b37973c2716bf5288d623
#owner: https://api.github.com/users/Dmytro-Pin

#Користувач з клавіатури вводить список цілих чисел. 
#Необхідно порахувати, скільки у списку парних і непарних чисел. 
#Результати вивести на екран.

num_list=[]
while True:
    user_inp=int(input('Введіть ціле число(для завершення введіть 0): '))
    if user_inp ==0:
        break
    num_list.append(user_inp)

even=0
for i in range(len(num_list)):
    if num_list[i]%2==0:
        even+=1
    continue

odd=0
for i in range(len(num_list)):
    if num_list[i]%2!=0:
        odd+=1
    continue

print('Кількість парних чисел: ', even)
print('Кількість непарних чисел: ', odd )
