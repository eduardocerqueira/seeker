#date: 2024-07-17T16:51:35Z
#url: https://api.github.com/gists/9661b1c5aa2eefe9562da9049e9bc6ea
#owner: https://api.github.com/users/MaXVoLD

def pass_word():
    list_value = []  # Список для хранения списков пар значений
    list_password = "**********"
    n = int(input('Введите число от 3 до 20: '))
    if 3 < n < 20:
        print('''
Введено неверное значние,
повторите попытку.
        ''')
        pass_word()
    else:
        for i in range(1, n):
            for j in range(1, n):
                if all([n % (i + j) == 0,
                        i != j,
                        [j, i] not in list_value]):
                    list_value.append([i, j])  # подобрал уникальные значения пар

    for i in list_value:
        list_password.extend(i)  # распаковал внутренние списки и объеденил их в один список

    password = "**********"
    print(password)


pass_word()
ки и объеденил их в один список

    password = "**********"
    print(password)


pass_word()
�ебрал значения из списка, присвоил их переменой
    print(password)


pass_word()
