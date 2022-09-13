#date: 2022-09-13T17:20:45Z
#url: https://api.github.com/gists/f71fcd0a08d8f0cc706028d0ffb99dcc
#owner: https://api.github.com/users/VictorZuyev

btcusd_exchange = int(input('What is Bitcoin price today?'))
own_money = int(input('How much $ do you have?'))
final_sum = round((own_money / btcusd_exchange), 7)
print('You can buy', final_sum, 'BTC')
