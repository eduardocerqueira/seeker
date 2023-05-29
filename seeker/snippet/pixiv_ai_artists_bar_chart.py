#date: 2023-05-29T16:57:28Z
#url: https://api.github.com/gists/fd11ccc3a078054082ff717b3411f94f
#owner: https://api.github.com/users/Tyhjakuori

# Data from https://github.com/Tyhjakuori/Personal-uBlock-filters/blob/main/pixiv_ai.txt

import matplotlib.pyplot as plt
import numpy as np

service_labels = [
    'No payservice links',
    'FANBOX',
    'Patreon',
    'Fantia',
    'DLsite',
    'Skeb',
    'pixiv Premium'
]

amounts = np.array([2027, 321, 300, 53, 50, 5, 861])

total_ids = 3617

percent = amounts/total_ids*100

new_labels = [i + ' {:.2f}%'.format(j) for i, j in zip(service_labels, percent)]

fig, ax = plt.subplots(layout='constrained')
colors = ['green', 'blue', 'purple', 'brown', 'teal', 'orange', 'pink']
plt.bar(service_labels, amounts, color=colors)
plt.xticks(range(len(service_labels)), new_labels)
plt.tight_layout()
ax.set_title('Payservice links AI artists pixiv', fontsize=20)
ax.set_xlabel('Payservice links', fontsize=20)
ax.set_ylabel('Amount', fontsize=20)

plt.show()
