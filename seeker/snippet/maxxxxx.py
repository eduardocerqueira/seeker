#date: 2022-09-14T17:24:56Z
#url: https://api.github.com/gists/e9a0d20431cd7ff4e3fd731cd58ffc42
#owner: https://api.github.com/users/MaxShulha94

bitcoin_rate = input('Enter bitcoin rate:')
dollars_sum = input('How much $ do you have?:')
bitcoin_rate = int(bitcoin_rate)
dollars_sum = int(dollars_sum)
sum = dollars_sum / bitcoin_rate
print ('You can buy', sum,'BTC')



