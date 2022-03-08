#date: 2022-03-08T17:08:41Z
#url: https://api.github.com/gists/a9cb79c584099778f0a9aa23eb38d85e
#owner: https://api.github.com/users/scorpions1340

import vk_api
from pyfiglet import Figlet
#после импорта нужной фигни, вносим токен  в переменную
token = "baf849a7801cc346aeb7685390f16e4ce6d1da42222bcbb5c15f259597cd0b0a86aaa4b2270459ff5b068"
#открываем диалог авторизации с vk, куда передадут токен
vk = vk_api.VkApi(token=token)
vk._auth_token()
#здесь a присваиваем метод из вк, в параметры указываем например users.search, так аналогично из доков хоть что
#тут по документации параметр q присваиваем имя таргета, так можно и фамилию и кучу всего, это для примера
#сортировка по популярным, offset-сдвиг поиска стоит нулевой, количество (count = 3), дальше доп. поле страны выведется
a = vk.method("users.search", {"q": "Александр",
                               "sort": 0, "offset": 0, "count": 3, "fields": "country", "city, sex": 1}).get("items")
#а дальше получаю именно items без остальной инфы, например так без этого:
#a = vk.method("users.search", {"q": "Александр",
                               #"sort": 0, "offset": 0, "count": 3, "fields": "country", "city, sex": 1})
#тут фигачим красивое оформление консольки со шрифтом slant, через метод renderText
preview = Figlet(font="slant")
print(preview.renderText("Hy Privet!"))
print(str(a))





