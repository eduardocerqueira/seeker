#date: 2022-11-15T16:58:34Z
#url: https://api.github.com/gists/e2e30a72a6deb54e702b673164f0d36f
#owner: https://api.github.com/users/DaSh-More

# цифры - фамиллии
people = {
    1: {'ru',  'fr', 'it'},
    2: {'fr', 'de'},
    3: {'ru', 'fr', 'de'},
    4: {'en', 'de', 'it'},
    5: {'en', 'it'},
}
common_langs = set()
non_rus_people = set()
needs_langs = {}

# Проходим по всем участникам
for n, man in enumerate(people.items()):
    name, langs = man
    # Если человек не знает русский
    if 'ru' not in langs:
        # Добавляем его в список
        non_rus_people.add(name)
        # Если он так же не знает английский
        if 'en' not in langs:
            # Проходим по языкам которые он знает
            for l in langs:
                # Добавляем каждый язык в словарь
                if l not in needs_langs:
                    needs_langs[l] = 0
    if n:
        common_langs &= langs
    else:
        # Если парвый раз идем по списку зададим начальный спиок языков
        common_langs = langs

# Еще раз проходим по языкам которые знают участники
for langs in people.values():
    # Увеличиваем количестов людей знающих язык
    for lang in langs:
        if lang in needs_langs:
            needs_langs[lang] += 1

print('Общие языки:', common_langs)
print('Участники не знающие русский:', non_rus_people)
print('Языки на которые так же нужен перевод:', needs_langs)
