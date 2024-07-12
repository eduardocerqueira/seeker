#date: 2024-07-12T16:43:12Z
#url: https://api.github.com/gists/7de081cf9ae2073b9603e77eadb242c3
#owner: https://api.github.com/users/MaXVoLD

students = {'Johnny', 'Bilbo', 'Steve', 'Khendrik', 'Aaron'}
grades = [[5, 3, 3, 5, 4], [2, 2, 2, 3], [4, 5, 5, 2], [4, 4, 3], [5, 5, 5, 4, 5]]
gpa = [sum(grades[0]) / len(grades[0]) ,
        sum(grades[1]) / len(grades[1]) ,
        sum(grades[2]) / len(grades[2]) ,
        sum(grades[3]) / len(grades[3]) ,
        sum(grades[4]) / len(grades[4]) ,] #Для каждого значения из списка нашел среднее значение. Составил новый список.
sorted_student = sorted(students) #Отсортировал имена по порядку A-Z. Вернул значения множества списком.
dict_gpa = dict(zip(sorted_student , gpa)) #Склеил оба списка между собой и объединил их в словарь.
print(dict_gpa)