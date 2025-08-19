#date: 2025-08-19T17:11:31Z
#url: https://api.github.com/gists/eaf720b69f7b37973c2716bf5288d623
#owner: https://api.github.com/users/Dmytro-Pin

#Користувач із клавіатури вводить список цілих чисел і деяке число. 
#Необхідно видалити зі списку всі елементи, які менші за задане число. 
#Результат вивести на екран.

num_list=[]
result=[]
while True:
    user_inp=int(input('Введіть ціле число(для завершення введіть 0): '))
    if user_inp ==0:
        break
    num_list.append(user_inp)

trigger=int(input('Введіть число для фільтрації: '))

for i in range(len(num_list)):
    if num_list[i]>trigger:
            result.append(num_list[i])
    continue

print('Кінцевий список: ')
for i in range(len(result)):
    print(result[i], end=', ')