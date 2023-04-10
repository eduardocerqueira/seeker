#date: 2023-04-10T16:42:42Z
#url: https://api.github.com/gists/42a9812bd5c663667015906061cbe818
#owner: https://api.github.com/users/MipoX

ip_address = []
while len(ip_address) != 4:
      len_ip = len(ip_address) + 1
      print('Введи {0}-ю часть ip bp 4-x чисел: '.format(len_ip), end= '')
      num_ip = int(input(''))
      if 0 < num_ip > 255:
            print('Указан неверный формат IP. IP не может превышать значения 255. ')
      else:
            ip_address.append(num_ip)


print('{}.{}.{}.{}'.format(ip_address[0], ip_address[1], ip_address[2], ip_address[3]))
