#date: 2022-05-02T16:57:17Z
#url: https://api.github.com/gists/05c42a81bcbb042e2a1b160da2b113ee
#owner: https://api.github.com/users/macndesign

def sum_hours(hours):
    total_secs = 0
    for tm in hours:
        time_parts = [int(s) for s in tm.split(':')]
        total_secs += (time_parts[0] * 60 + time_parts[1]) * 60

    total_secs, _ = divmod(total_secs, 60)
    hr, min = divmod(total_secs, 60)
    return "%d:%02d" % (hr, min)


print('Press "e" to exit')

time_list = []

while True:
    time = input('Enter time (HH:MM): ')
    if time == 'e':
        break
    time_list.append(time)

print(sum_hours(time_list))
