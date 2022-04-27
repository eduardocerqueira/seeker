#date: 2022-04-27T17:20:44Z
#url: https://api.github.com/gists/e7a2f3bf4ca199d34c1e0e9dcb0c6515
#owner: https://api.github.com/users/mubashariqbal

#!/usr/bin/python
# coding=utf-8
#
# <xbar.title>Deso Ticker</xbar.title>
# <xbar.version>v1.0</xbar.version>
# <xbar.author>mubashariqbal</xbar.author>
# <xbar.author.github>mubashariqbal</xbar.author.github>
# <xbar.desc>Displays current Deso price from Coingecko</xbar.desc>
# <xbar.image></xbar.image>
#
# by mubashariqbal

from urllib import urlopen
url = urlopen('https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=bitclout').read()

up = "image=iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QAyQACAALwzISXAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH4AQHACkSBTjB+AAAALNJREFUOMvVk70NAjEMhb87WYiGBZAQU7ABNSVSWpZgEEagsJDoKBELUCEKFuBuCKTw0xyQC0lICe5i+/k9/wT+3opUUJQhcAUqa8I5ZQT4tANwioGTCkQZA9vmOQE2oUJFhL0DXBz33RpKUfCLfLTQJMx9IlEWuQr6QB3prGtNS1lwiMvEYo7ekNsKRBkB+y+rH1hDFVOwy7ids+gbVzrsM6CXeYDTF85xroB1ZoHb73ymB5RhJkpZTihGAAAAAElFTkSuQmCC"
down = "image=iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QABACnAADQ9FZaAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH4AQHACQ1FZwK3gAAAMRJREFUOMvNkjEKAjEQRZ+jKNjYKh5AbzCdjVcQj+BFPIKlp7EMeAJrUbASQVCEr80uG9cNbqe/Cgn/5WUI/DqNfBHM+kCzbs+lPUAr2pwBq5qABbB+M8gszkDvS/kOdAG5VBgEM4ApsP0CGLukjxlEoA0wSZR3Lo0qhxhZDIBDAmDA0wsBLD51CZeOwLKivHbprZx6AkAHuEXbD5fawYwywMqAzOKeDTTPvKqcTGZBMLsGs0utn5gADYEHcKp9e9ni//MCDtNCE3qjsIwAAAAASUVORK5CYII="
no_change = ""

deso = "image=iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAIqSURBVHgBTVPLahRBFD23qnq6W2ecJIgkwQeKCuJGgoi4cOdGv8BlyMZP8Ftc6W+4FbKQIBiRbJKIwRmNJIGJznTidNf11qMfBQVVt+pUnXvuuYR1y0QEN5gYYRn2MQwodxbCLsZ1XO4bdyghf6IaRBtzIFadhyLQ/yHnRilqAFcuE26vWGSGkBqgZxipBhLN+PabsblPsFSzo8igZiyB4ZCxvEq4tWBxZ0lhkAKXZA56bs3YO7bYeAdMCuXBJJ8r1oEiU9DgvCLsHGt8OpS1JSRyliiWSXh0VeHV01JSCeCQVRTH5TcWmicTi548enpO+HBgsXNUobTsY0YTntxQoI5W7QMyp2cKm18I27sMuY/FTOGoMNj6RRhP2TNJkwiOnxp0SsIUxNkdA3vjEg9vAs8fOCET7J8wpkWFYh4qY1UokmHV1p7qEslYyBkv7gEbawqjPxUOZwStNQ4mFZxudfEMNSkE4MXM4uUasP5YY3mgBARcG1okQr0SgV1VSD7laLRopED9/qrF62cW15eMeAAoSsbfOWMuwEz2WY/RT1XrUmekrqJ3Vwj9CwZnJXzZ/lUCEgb9nJCLmZzB3n8tBJh7MHsGnWGtizMWBTBIRDxDHpRHR775WODt57QBU+PEWJat7+KDGXuB3NRyQWv26W2PSvycpWJl5W1fN5WpM3Cb0UThx2mnC2uBfWP1Yjdye+aq4C3sLRV+ahRR9UVHt24gjt0YRHcs/wPGurdlxWoyAQAAAABJRU5ErkJggg=="

import json
result = json.loads(url)

pct = result[0]['price_change_percentage_24h']
change = result[0]['price_change_24h']
last = result[0]['current_price']

def flow():
    if pct > 0:
        print ('$%.2f +%.2f%% | %s | color=green '% (float(last), float(pct), deso))
    elif pct < 0:
        print ('$%.2f %.2f%% | %s | color=red '% (float(last), float(pct), deso))
    else:
        print ('$%.2f %.2f%% | %s | color=gray '% (float(last), float(pct), deso))

flow()