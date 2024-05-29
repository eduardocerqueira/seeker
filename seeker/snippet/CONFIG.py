#date: 2024-05-29T16:58:21Z
#url: https://api.github.com/gists/853f74403aed8c7efe6b30e7195d71c1
#owner: https://api.github.com/users/adepte-fufayka

TOKEN='7350631593: "**********"

class Place:
    def __init__(self, name, _type, found, zone, x, y):
        self.name = name
        self._type = _type
        self.found = found
        self.zone = zone
        self.x = x
        self.y = y

    def __str__(self):
        return f'{self.name}\n{self._type}\n{self.found}\n{self.zone}\n{self.x}\n{self.y}\n'

class User:
    def __init__(self, uid, username, name, squad_name, time, res_time, deff, attack, health_p, power_p, mana_p, role,
                 boss_ping, city, prof, prof_time, timezone):
        self.uid = uid
        self.city = city
        self.res_time = res_time
        self.deff = deff
        self.attack = attack
        self.username = username
        self.name = name
        self.squad_name = squad_name
        self.time = time
        self.mana_p = mana_p
        self.power_p = power_p
        self.health_p = health_p
        self.role = role
        self.boss_ping = boss_ping
        self.prof = prof
        self.prof_time = prof_time
        self.timezone = timezone

    def __str__(self):
        return f'{self.uid}\n{self.username}\n{self.name}\n{self.squad_name}\n{self.time}\n{self.res_time}\n{self.deff}\n{self.attack}\n{self.health_p}\n{self.power_p}\n{self.mana_p}\n{self.role}\n{self.boss_ping}\n{self.city}\n{self.prof}\n{self.prof_time}\n{self.timezone}\n'
e}\n{self.timezone}\n'
