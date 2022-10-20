#date: 2022-10-20T17:23:27Z
#url: https://api.github.com/gists/69145d669c687e0b936cb51f6ee3d74f
#owner: https://api.github.com/users/rmorenobello

import matplotlib.pyplot as plt  # we only import pyplot

month_number = [1, 2, 3, 4, 5, 6, 7]
new_deaths = [213, 2729, 37718, 184064, 143119, 136073, 165003]

plt.plot(month_number, new_cases)
plt.ticklabel_format(axis='y', style='plain')
plt.title('New Reported Cases By Month (Globally)')
plt.xlabel('Month Number')
plt.ylabel('Number Of Cases')
plt.show()