#date: 2022-10-17T17:21:36Z
#url: https://api.github.com/gists/a71e45574279c4ae75abaaf5f49153f4
#owner: https://api.github.com/users/LowderPlay

vowels = {
    'а': {
        'double': None,
        'softener': False
    },
    'о': {
        'double': None,
        'softener': False
    },
    'у': {
        'double': None,
        'softener': False
    },
    'ы': {
        'double': None,
        'softener': False
    },
    'э': {
        'double': None,
        'softener': False
    },

    'я': {
        'double': 'йа',
        'softener': True
    },
    'ё': {
        'double': 'йо',
        'softener': True
    },
    'ю': {
        'double': 'йу',
        'softener': True
    },
    'и': {
        'double': None,
        'softener': True
    },
    'е': {
        'double': 'йэ',
        'softener': True
    }
}

consonants = {
    'н': {
        'soft': None,
        'voiced': True
    },
    'м': {
        'soft': None,
        'voiced': True
    },
    'л': {
        'soft': None,
        'voiced': True
    },
    'р': {
        'soft': None,
        'voiced': True
    },
    'й': {
        'soft': True,
        'voiced': True
    },
    'б': {
        'soft': None,
        'voiced': True
    },
    'в': {
        'soft': None,
        'voiced': True
    },
    'г': {
        'soft': None,
        'voiced': True
    },
    'д': {
        'soft': None,
        'voiced': True
    },
    'ж': {
        'soft': False,
        'voiced': True
    },
    'з': {
        'soft': None,
        'voiced': True
    },


    'п': {
        'soft': None,
        'voiced': False
    },
    'ф': {
        'soft': None,
        'voiced': False
    },
    'к': {
        'soft': None,
        'voiced': False
    },
    'т': {
        'soft': None,
        'voiced': False
    },
    'ш': {
        'soft': None,
        'voiced': False
    },
    'с': {
        'soft': None,
        'voiced': False
    },
    'х': {
        'soft': None,
        'voiced': False
    },
    'ц': {
        'soft': None,
        'voiced': False
    },
    'ч': {
        'soft': None,
        'voiced': False
    },
    'щ': {
        'soft': None,
        'voiced': False
    },
}

word = input()
sounds = []
for i, letter in enumerate(word):
    sound = {
        'symbol': letter,
        'sound': ''
    }
    if letter in consonants:
        sound['type'] = 'consonant'
        sound['voiced'] = consonants[letter]['voiced']
        sound['sound'] = letter
        if consonants[letter]['soft'] is not None:
            sound['soft'] = consonants[letter]['soft']
        else:
            sound['soft'] = False
            if len(word) > i:
                if word[i + 1] in ['ь'] + list(filter(lambda k: vowels[k]['softener'], vowels.keys())):
                    sound['soft'] = True
        if sound['soft']:
            sound['sound'] += '\''
    elif letter in vowels:
        sound['type'] = 'vowel'
        sound['sound'] = letter
        if vowels[letter]['double'] is not None:
            if i == 0 or letter in ['я', 'ю', 'ё']:
                sound['sound'] = vowels[letter]['double']
            else:
                if word[i - 1] in ['ъ', 'ь'] + list(vowels.keys()):
                    sound['sound'] = vowels[letter]['double']
                elif letter == 'е':
                    sound['sound'] = 'э'

    else:
        sound['type'] = ''
    sounds.append(sound)
print(sounds)
