#date: 2021-09-03T17:14:11Z
#url: https://api.github.com/gists/4594a46860bce1018353c5f615ebefc3
#owner: https://api.github.com/users/Maine558

n, m = map(int, input().split())

RGB = []
power = []

for i in range(n):
    RGB.append([str(s) for s in input().split()])

for i in range(n):
    power.append([str(s) for s in input().split()])

numbers = [
    ["."],
    [".", "B"],
    [".", "G"],
    [".", "G", "B"],
    [".", "R"],
    [".", "R", "B"],
    [".", "R", "G"],
    [".", "R", "G", "B"]
]

for i in range(len(power)):
    for j in range(len(power[i])):
        for z in range(len(numbers)):
            if power[i][j] == str(z):
                power[i][j] = numbers[z]
print(RGB[0])
print(power[0])

# Смотрим сколько у нас символов в RGB подмассиве - сравниваем их кол-во с подмассивом power
# Если их кол-во равно, то заменяем все клетки, иначе смотрим другой символ
# Пример: [RGGB.] и [["."],[".","R","G"],[".","G","B"],[".","G","B"],[".","B"]] - Кол-во красных одинаков -
# [RGGB.] - [["."],R,[".","G","B"],[".","G","B"],[".","B"]] - сортируем дальше. Одинаковое кол-во G в массивах -
# [RGGB.] - [["."],R,G,G,[".","B"]] - дальше по алгоритму
# [RGGB.] - [.,R,G,G,B] - итог, да, можно, но если их больше, то заменять тех, что меньше
# и анализировать дальше
# в ином случае просто выводить - No
#