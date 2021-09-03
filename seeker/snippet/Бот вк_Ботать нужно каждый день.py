#date: 2021-09-03T17:13:52Z
#url: https://api.github.com/gists/0c517b21dcfe9cc8f906d3f06156d58b
#owner: https://api.github.com/users/Maine558

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
import datetime



vk_session = vk_api.VkApi(token="80ae1d54376410ff0be5ea3f9767c1a53f7389c264064fdae9ed6c6bc3fc7065c125276e238a38f8ea954")

session_api = vk_session.get_api()

longpoll = VkLongPoll(vk_session)



time = datetime.datetime.today().strftime("%H:%M:%S")
def sender(id,text):
    vk_session.method("messages.send",{"user_id" : id,"message" : text, "random_id" : 0})

New = True


for event in longpoll.listen():
    if event.type == VkEventType.MESSAGE_NEW:
        if event.to_me:
            msg = event.text.lower()
            id = event.user_id

            if New == True:
                sender(id,"Да-да-да или нет-нет-нет. Погоди. Сейчас я тебе расскажу, что мне подвластно! Во-первых, это напоминания о том, что нужно ботать.")
                New = False
