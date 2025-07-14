#date: 2025-07-14T17:00:09Z
#url: https://api.github.com/gists/8da80ad472aeb19be259b3a59ac95da2
#owner: https://api.github.com/users/mustafaadogan

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib as mpl
mpl.font_manager.fontManager.addfont('IBMPlexSans-Regular.ttf') # https://fonts.google.com/specimen/IBM+Plex+Sans
plt.rcParams['font.family'] = 'IBM Plex Sans'
plt.rcParams['font.weight'] = 'regular'

# Set theme
sns.set_theme(style="whitegrid")

# Your data
data = {
    "Settings": ["0-Shot", "4-Shot\nRandom", "4-Shot\nSimilar", "8-Shot\nSimilar", "8-Shot\nSimilar+CoT"],
    "Qwen2.5-VL-3B (I)": [75.27, 78.98, 79.07, 78.42, 73.98],
    "Qwen2.5-VL-7B (I)": [81.27, 82.67, 83.93, 84.51, 82.13],
    "Qwen2-VL-2B": [18.40, 62.18, 63.53, 61.20, 54.16],
    "Qwen2-VL-2B (I)": [65.53, 68.33, 64.82, 64.33, 65.22],
    "Qwen2-VL-7B": [36.18, 79.11, 77.89, 78.51, 77.60],
    "Qwen2-VL-7B (I)": [75.98, 80.22, 80.27, 79.98, 78.20],
    "InternVL2.5-4B-MPO (I)": [77.40, 74.02, 72.91, 71.00, 52.38],
    "InternVL2.5-8B-MPO (I)": [81.62, 79.80, 78.96, 77.73, 76.69],
    "InternVL2.5-4B (I)": [73.47, 73.24, 72.22, 70.09, 45.96],
    "InternVL2.5-8B (I)": [76.91, 77.27, 74.89, 75.71, 69.53],
    "Idefics3-8B (I)": [68.36, 72.22, 72.29, 72.44, 71.09],
    "Idefics2-8B (I)": [67.76, 68.33, 68.73, 69.04, 65.76],
    "Phi-4-Multimodal (I)": [71.87, 66.62, 67.36, 66.62, 62.98],
    "Phi-3.5-Vision (I)": [63.07, 59.33, 59.84, 58.51, 53.78],
    "LLaVA-Interleave-Qwen-7b (I)": [74.76, 68.04, 65.67, 66.22, 63.73],
    "LLaVA-Interleave-Qwen-0.5b (I)": [51.00, 46.20, 44.16, 43.60, 35.29],
    "xGen-MM-Phi3-Mini-R-v1.5 (I)": [70.00, 68.82, 67.91, 68.42, 59.51],
    "xGen-MM-Phi3-Mini-Base-R-v1.5": [15.84, 38.44, 36.82, 36.53, 30.31]
}

data2 = {
    "Qwen2.5-VL-3B (I)": [75.27, 78.98, 79.07, 78.42, 73.98],
    "Qwen2.5-VL-7B (I)": [81.27, 82.67, 83.93, 84.51, 82.13],
    "Qwen2-VL-2B": [18.40, 62.18, 63.53, 61.20, 54.16],
    "Qwen2-VL-2B (I)": [65.53, 68.33, 64.82, 64.33, 65.22],
    "Qwen2-VL-7B": [36.18, 79.11, 77.89, 78.51, 77.60],
    "Qwen2-VL-7B (I)": [75.98, 80.22, 80.27, 79.98, 78.20],
    "InternVL2.5-4B-MPO (I)": [77.40, 74.02, 72.91, 71.00, 52.38],
    "InternVL2.5-8B-MPO (I)": [81.62, 79.80, 78.96, 77.73, 76.69],
    "InternVL2.5-4B (I)": [73.47, 73.24, 72.22, 70.09, 45.96],
    "InternVL2.5-8B (I)": [76.91, 77.27, 74.89, 75.71, 69.53],
    "Idefics3-8B (I)": [68.36, 72.22, 72.29, 72.44, 71.09],
    "Idefics2-8B (I)": [67.76, 68.33, 68.73, 69.04, 65.76],
    "Phi-4-Multimodal (I)": [71.87, 66.62, 67.36, 66.62, 62.98],
    "Phi-3.5-Vision (I)": [63.07, 59.33, 59.84, 58.51, 53.78],
    "LLaVA-Interleave-Qwen-7b (I)": [74.76, 68.04, 65.67, 66.22, 63.73],
    "LLaVA-Interleave-Qwen-0.5b (I)": [51.00, 46.20, 44.16, 43.60, 35.29],
    "xGen-MM-Phi3-Mini-R-v1.5 (I)": [70.00, 68.82, 67.91, 68.42, 59.51],
    "xGen-MM-Phi3-Mini-Base-R-v1.5": [15.84, 38.44, 36.82, 36.53, 30.31]
}

df = pd.DataFrame(data)
df = df.melt(id_vars="Settings", var_name="Model", value_name="Score")

# Map model to family
def get_family(model_name):
    if model_name.startswith("Qwen"):
        return "Qwen"
    elif model_name.startswith("InternVL"):
        return "InternVL"
    elif model_name.startswith("Phi"):
        return "Phi"
    elif model_name.startswith("Idefics"):
        return "Idefics"
    elif model_name.startswith("LLaVA"):
        return "LLaVA"
    elif model_name.startswith("xGen"):
        return "xGen"
    return "Other"

# Identify tuning
def get_linestyle(model_name):
    return '-' if '(I)' in model_name else '--'

# Assign a unique marker per model
markers = ['o', 's', 'D', 'p', 'v', '^']
model_list = df['Model'].unique()
model_marker_map = {}
ii = 0
iii = 0
for model_name in data2.keys():
    model_marker_map[model_name] = markers[ii % len(markers)]

    if iii in [5, 9, 11, 13, 15]:
        ii = -1

    iii += 1
    ii += 1

# Color per family
family_color = {
    "Qwen": "#1f77b4",
    "InternVL": "#ff7f0e",
    "Phi": "#2ca02c",
    "Idefics": "#d62728",
    "LLaVA": "#9467bd",
    "xGen": "#8c564b"
}

# Plotting
plt.figure(figsize=(12, 7))

for model in model_list:
    model_df = df[df['Model'] == model]
    family = get_family(model)
    linestyle = get_linestyle(model)
    marker = model_marker_map[model]
    plt.plot(
        model_df['Settings'], model_df['Score'],
        label=model,
        color=family_color[family],
        linestyle=linestyle,
        marker=marker
    )

# Labels and Legend
plt.xlabel("Settings")
plt.ylabel("Accuracy (%)", fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=30, fontsize=11)
plt.yticks(fontsize=11)

# Legend to center right
plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0., fontsize=9, title="Models")

plt.tight_layout()
plt.savefig("few_shot_comparison_fancy_family_color.pdf", dpi=300, bbox_inches='tight')
plt.show()