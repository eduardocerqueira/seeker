#date: 2025-09-09T17:13:47Z
#url: https://api.github.com/gists/f592f11d6c2c4079650eb533a9b355ff
#owner: https://api.github.com/users/dartmew

from transliterate import translit
from num2words import num2words

print(translit('''Ladies and gentlemen, I'm 78 years old and I finally got 15 minutes of fame once in a lifetime and I guess that this is mine. People have also told me to make these next few minutes escruciatingly embarrassing and to take vengeance of my enemies. Neither will happen.

More than 3 years ago I moved to Novo-Novsk, but worked on new Magnetic Storage for last 40. When I was 8...''', 'ru'))

print('78', translit(num2words(78), 'ru'), sep = ' - ')
print('15', translit(num2words(15), 'ru'), sep = ' - ')
print('3', translit(num2words(3), 'ru'), sep = ' - ')
print('40', translit(num2words(40), 'ru'), sep = ' - ')
print('8', translit(num2words(8), 'ru'), sep = ' - ')