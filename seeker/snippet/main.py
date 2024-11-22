#date: 2024-11-22T17:05:46Z
#url: https://api.github.com/gists/393fb7bbca0eaf3526b433693a6687d1
#owner: https://api.github.com/users/crypt0xCode

"""
Уровень 1:
1) Температура воды в кастрюле 7 градусов. Кастрюлю поставили на огонь. Температура увеличивается каждые 2с на 1 градус. Определить через сколько вода закипит (температура будет 100 градусов)
2) Вывести на экран 5 строк из трех нулей (каждая строка должна быть пронумерована)
3) Вывести прямоугольный треугольник из символа "*" (высота указывается с клавиатуры)
"""
def start_level1():
    temp: int = 7
    time: int = 0
    while(temp <= 100):
        time += 2
        # Жаль, что нельзя temp++ =)
        temp += 1
    print(f'Temperature will be over 100 after {time} seconds')

    for i in range(0, 5):
        print(f'{i}. 000')

    height: int = int(input('Enter height: '))
    width = height
    step: int = 2
    i: int = 0
    j: int = 0
    while(i < height):
        # Change rows.
        if (i == height-1):
            for k in range(0, width):
                for n in range(0, step-1):
                    print('*', end='')
                    print(' ', end='')
        else:
            # Print columns.
            for m in range(0, j+1):
                if (m == 0):
                    print('*', end='')
                elif (m == j):
                    print('*', end='')
                else:
                    print(' ', end='')
        i += 1
        j += step
        print()


"""
Уровень 2:
1) Имеется коробка со сторонами A x B x C. Определить, войдет ли она в дверь размером M x K
2) Вывести равнобедренный треугольник из символа "*" (высота указывается с клавиатуры)
3) Дано число N и последовательность квадратов чисел (1, 4, 9, 16, 25, …). Вывести числа, которые меньше N
"""
def start_level2():
    a: int = int(input('Enter a: '))
    b: int = int(input('Enter b: '))
    c: int = int(input('Enter c: '))
    box = [a, b, c]

    m: int = int(input('Enter m: '))
    k: int = int(input('Enter k: '))
    door = [m, k]

    if sum(box) <= 2 * sum(door):
        print('Pass.')
    else:
        print('You shall not pass!')

    height: int = int(input('Enter height: '))
    width = height
    step: int = 2
    start_point: int = width
    i: int = 0
    j: int = 0
    while (i < height):
        # Set draw point.
        for p in range(0, start_point):
            if (p == start_point - 1):
                # Change rows.
                if (i == height - 1):
                    for k in range(0, width):
                        for n in range(0, step - 1):
                            print('*', end='')
                            print(' ', end='')
                else:
                    # Print columns.
                    for m in range(0, j + 1):
                        if (m == 0):
                            print('*', end='')
                        elif (m == j):
                            print('*', end='')
                        else:
                            print(' ', end='')
                i += 1
                j += step
            else:
                print(' ', end='')
        start_point -= 1
        print()

    n: int = int(input('Enter n: '))
    squares = [x ** 2 for x in range(0, 101)]
    less_numbers = list()
    for i in squares:
        if n > i:
            less_numbers.append(i)
    print(f'There are {len(less_numbers)} numbers less than {n}:')
    print(less_numbers)


"""
Уровень 3:
1) В кафе продают по 3 шарика мороженного и по 5. Можно ли купить ровно k шариков мороженного
2) Клиент оформил вклад на m тысяч рублей в банке под k% годовых. Через сколько лет сумма вклада превысит s тысяч рублей, если за это время клиент не будет брать деньги со счета. 
3) Дано число N. Посчитать сумму цифр
"""
def start_level3():
    icecream_balls: int = int(input('Enter icecream balls amount: '))
    if icecream_balls % 3 == 0 or icecream_balls % 5 == 0:
        print('Yes.')
    elif icecream_balls < 3 or icecream_balls < 5:
        print('No.')
    else:
        sum: int = 0
        while(sum < icecream_balls):
            if ((icecream_balls - sum) % 3 == 0):
                sum += 3
            else:
                sum += 5

        if sum == icecream_balls:
            print('Yes.')
        else:
            print('No.')

    deposit: float = float(input('Enter deposit in RUB: '))
    percents: int = int(input('Enter percents: '))
    ages: int = 0
    awaiting_sum: float = float(input('Enter awaiting sum in RUB: '))

    while(deposit <= awaiting_sum):
        deposit += (deposit * percents) / 100
        ages += 1
    print(f'Awaiting sum {awaiting_sum} RUB will be collected after {ages} years with {percents}%')

    n: str = input('Enter n: ')
    digits = [int(item) for item in n]
    print(f'Sum of {n} digits is {sum(digits)}')


def main():
    # Uncomment for starting specify level.
    # start_level1()
    # start_level2()
    start_level3()
if __name__ == '__main__':
    main()