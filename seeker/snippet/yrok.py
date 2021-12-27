#date: 2021-12-27T16:56:02Z
#url: https://api.github.com/gists/503442bc2e956a72c03accd3c084b54d
#owner: https://api.github.com/users/travvva

# Дан файл с пунктами меню (id, название, id родителя).
# Если id родителя равно 0, то родителя не существует. Показать полное меню с отступами.
# Пользователь вводит id пункта. Показать цепочку из пунктов меню до этого пункта.
# Уровней вложенности в меню может быть любое количество.
items = []


# открыли файл и выводим все данные файла
with open('cats.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line[:line.index(' ')].isdigit() and \
                line[line.rindex(' ') + 1:].isdigit():
            items.append({
                'id': int(line[:line.index(' ')]),
                'title': line[line.index(' '):line.rindex(' ') + 1].strip(),
                'id_parent': int(line[line.rindex(' ') + 1:])
            })
print(items)
#  выводим весь файл с отступами
def print_item_on_id(items, item, deep, in_id):
    print('\t' * deep, item['title'])
    if not item['id'] == in_id:
        for num in range(len(items)):
            if items[num]['id_parent'] == item['id']:
                print_item_on_id(items, items[num], deep + 1, in_id)
    else:
        exit()


# если родитель 0, то родителя нет
for num in range(len(items)):
    if items[num]['id_parent'] == 0:
        print_item_on_id(items, items[num], 0, 0)


# вводим айди и выводим текст до указанного айди
id = int(input('Введите id: '))
for num in range(len(items)):
    if items[num]['id_parent'] == 0:
        print_item_on_id(items, items[num], 0, id)
