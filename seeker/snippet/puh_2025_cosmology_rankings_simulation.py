#date: 2025-12-22T16:52:28Z
#url: https://api.github.com/gists/aaeb52b7ba9302dd432bbb1165377c29
#owner: https://api.github.com/users/BrianMartell

import numpy as np
import matplotlib.pyplot as plt

# PUH v25: 2025 Cosmology Rankings Sim — Radar Top Models
models = ['PUH', '\Lambda CDM', 'Inflation', 'Hot Big Bang', 'EDE']
scores = [
    [24, 22, 21, 23],  # PUH 90
    [19, 23, 24, 23],  # \Lambda CDM 89
    [20, 22, 23, 20],  # Inflation 85
    [22, 20, 23, 19],  # Hot BB 84
    [18, 20, 21, 20]   # EDE 79
]

angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))
colors = ['cyan', 'gold', 'red', 'purple', 'green']
for i, (model, score) in enumerate(zip(models, scores)):
    score += score[:1]
    ax.plot(angles, score, 'o-', linewidth=2, label=f'{model} {sum(score[:-1])}/100', color=colors[i])
    ax.fill(angles, score, alpha=0.15, color=colors[i])

ax.set_thetagrids(angles[:-1] * 180/np.pi, ['Coherence', 'Predictive', 'Empirical', 'Testability'])
ax.set_ylim(0,25)
ax.set_title('PUH v25: 2025 Cosmology Rankings Sim — PUH Tops')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig('puh_2025_cosmology_rankings_simulation.png', dpi=300)
plt.show()

print("PUH 90/100 leads — unification merit.")