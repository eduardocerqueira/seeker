#date: 2022-09-13T17:16:44Z
#url: https://api.github.com/gists/4b7c7d5c0b2fc58229078197d11bd4db
#owner: https://api.github.com/users/olegveisa

price_btc = int(input())
print('What is Bitcoin price today?', price_btc, sep = '\n')

money = int(input())
print('How much $ do you have?', money, sep = '\n')

btc = money / price_btc
print(f"You can buy {btc} BTC")