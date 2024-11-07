#date: 2024-11-07T17:12:10Z
#url: https://api.github.com/gists/32d558682b62350b1b703d1965c7487e
#owner: https://api.github.com/users/IndraReddy5

def convert_into_float(price):
    if isinstance(price, float) or isinstance(price, int):
        return float(price)
    if price.isnumeric():
        return float(price)
    else:
        t = ''
        for i in price:
            if i.isnumeric():
                t += i
        price = float(t)
        return price