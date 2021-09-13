#date: 2021-09-13T17:17:15Z
#url: https://api.github.com/gists/17185c052d913055aa19cfeda0db58a9
#owner: https://api.github.com/users/gustavomotadev

def price_br_to_int(price):
    price_int, price_dec = price.split(',', 1)
    price_int = price_int.split('.')
    return ''.join(price_int) + '.' + price_dec
    
print(price_br_to_int('1.010,90'))