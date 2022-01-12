#date: 2022-01-12T17:16:12Z
#url: https://api.github.com/gists/46c59e6e97800f14cab6c816a3106a61
#owner: https://api.github.com/users/nerudesu

total_hashrate = 0
for miner in miners_data:
    total_hashrate = total_hashrate + miner['hashrate']

print(total_hashrate)


def count_digit(number):
    if number == 0:
        return 1
    return int(log10(number)) + 1


def display_hashrate(hashrate):
    hashrate_digit = count_digit(hashrate)

    if hashrate_digit > 3:
        return round(hashrate / 1000, 2), "KH/s"
    elif hashrate_digit >= 6:
        return round(hashrate / 1000000, 2), "MH/s"
    else:
        return round(hashrate, 2), "H/s"


hash, order = display_hashrate(total_hashrate)
print(hash, order, "(5)")