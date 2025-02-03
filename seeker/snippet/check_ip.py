#date: 2025-02-03T17:01:58Z
#url: https://api.github.com/gists/0da0c0de531502df1a5504f1672c64e7
#owner: https://api.github.com/users/k3vmars

from ipwhois import IPWhois

# Список IP-адресов
ip_addresses = [
    "37.140.192.105",
    "104.236.55.36",
    "118.27.100.90",
]

# Функция для проверки, принадлежит ли IP регистратору "reg.ru"
def check_ip(ip):
    ipwhois = IPWhois(ip)
    result = ipwhois.lookup_rdap()

    org_name = result.get('network', {}).get('name', 'Не найдено')

    if 'REGRU' in org_name:
        # Выводим информацию только если IP принадлежит "reg.ru"
        print(f"IP {ip} принадлежит reg.ru")
        return True
    return False

# Проверка каждого IP-адреса
for ip in ip_addresses:
    check_ip(ip)  # Если IP принадлежит "reg.ru", будет выведено сообщение