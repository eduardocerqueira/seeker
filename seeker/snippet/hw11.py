#date: 2024-10-29T16:47:42Z
#url: https://api.github.com/gists/87151edc60d0c7e08102ec1fb63eefc9
#owner: https://api.github.com/users/JourneymanFO

def calc(a, simbol, b):
    try:
        a = float(a)
        b = float(b)
        if simbol in ['+', '-', '*', '/', '//', '**', '%']:
            if simbol == '+':
                return a + b
            elif simbol == '-':
                return a - b
            elif simbol == '*':
                return a * b
            elif simbol == '/':
                return a / b
            elif simbol == '**':
                return a ** b
            elif simbol == '%':
                return a % b
            elif simbol == '//':
                return a // b
        else:
            return 'Вы ввели не корректные символы!'
    except:
        return 'Нужно ввести число!'

a = (input('Введите первое число: '))
simbol = str(input('Введите знак (+, -, /, *): '))
b = (input('Введите второе число: '))
c = calc(a, simbol, b)
print(f'{a} {simbol} {b} = {c}')


def quiz():
    counter = 0
    number_of_questions = 5
    percents = 100 / number_of_questions
    
    questions = {
        'Сколько цветов в радуге?': ['a. 2', 'b. 23', 'c. 7'],
        'Ты сейчас смотришь в монитор?': ['a. Да', 'b. Нет', 'c. Не знаю'],
        'Верно ли вычисление 2+2=5?': ['a. Да', 'b. Нет', 'c. Не знаю'],
        'Чему равно число Пи?': ['a. 2.13', 'b. 3.31', 'c. 3.14'],
        'Кто основатель Братства Стали?': ['a. Паладин Ромбус', 'b. Роджер Мэксон', 'c. Выходец из Убежища'],
    }
    correct_answers = {
        'Сколько цветов в радуге?': 'c',
        'Ты сейчас смотришь в монитор?': 'a',
        'Верно ли вычисление 2+2=5?': 'b',
        'Чему равно число Пи?': 'c',
        'Кто основатель Братства Стали?': 'b',
    }

    for question, options in questions.items():
        print(f'\n{question}\nВарианты ответов: {options}')
        
        answer = input('Напишите вариант ответа (a/b/c): ').strip().lower()
        
        if answer in ['a', 'b', 'c']:
            if answer == correct_answers[question]:
                counter += 1
                print('Верный ответ!\n')
            else:
                print('Неверный ответ.\n')
        else:
            print('Некорректный ввод. Введите только a, b или c.\n')
            break

    score = counter * percents
    if counter > 0:
        print(f'Вы завершили квиз на {score}%')
    else: 
        print('Вы не ответили ни на один вопрос :(')

print('Добро пожаловать на квиз из 5 вопросов!')
start = input('Начнём? (Yes/No): ').strip().lower()

while start == 'yes':
    quiz()
    start = input('Хотите пройти квиз снова? (Yes/No): ').strip().lower()
    if start != 'yes':
        print('Спасибо за участие в квизе!')
        break
