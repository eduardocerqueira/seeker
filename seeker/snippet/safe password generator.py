#date: 2023-06-14T16:59:41Z
#url: https://api.github.com/gists/f9df155e8fe3df55f124a7ec13b49f6e
#owner: https://api.github.com/users/nabocovian

# безопасный пароль
import random

#приветствие
name = input('Привет - ').title()
print(f'Вижу, вы, {name}, в поисках надежного пароля - вы по адресу!')

#константы для пароля
digits = '1234567890'
lowercase_letters = 'abcdefghijklmnopqrstuvwxyz'
uppercase_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
punctuation = '!#$%&*+-=?@^_'

chars = ''

#функция введено число
def is_valid(num):
    if num.isdigit():
        return True
    else:
        return False

#функция определения "да"

def is_da(txt):
    if txt.lower() in ['да', 'д']:
        return True
    else:
        return False

#вопросы к паролю
print('Уточним кое-что по вашему паролю:')
q1 = input('Сколько паролей необходимо?')
if is_valid(q1):
    q1 = int(q1)
else:
    print('Что-то пошло не так...')
q2 = input('Какая длина каждого пароля?')
if is_valid(q2):
    q2 = int(q2)
else:
    print('Что-то пошло не так...')
q3 = input('Использовать ли цифры?')
if is_da(q3):
    chars += digits
q4 = input('А прописные буквы?')
if is_da(q4):
    chars += uppercase_letters
q5 = input('Использовать строчные буквы?')
if is_da(q5):
    chars += lowercase_letters
q6 = input('А символы?')
if is_da(q6):
    chars += punctuation
q7 = input('Что насчет неоднозначных символов?')
if is_da(q7):
    for _ in 'il1Lo0O':
        chars = chars.replace(_, '')

#функция генератор пароля
 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"e "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"( "**********"l "**********"e "**********"n "**********"g "**********"h "**********"t "**********", "**********"  "**********"c "**********"h "**********"a "**********"r "**********"s "**********") "**********": "**********"
    password = "**********"
    for i in range(lenght):
        password += "**********"
    return password

for i in range(q1):
    print(generate_password(q2, chars))

print(f'Вот такие клевые пароли у нас получились, {name},забирай!')
print('Если понадоблюсь - приходи еще! До встречи!')
