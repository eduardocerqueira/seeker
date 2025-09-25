#date: 2025-09-25T16:58:25Z
#url: https://api.github.com/gists/62aafa90c13d6f127bd6f43632c501c2
#owner: https://api.github.com/users/jdwebprogrammer

import pandas as pd
import matplotlib.pyplot as plt

# List of SI prefixes with symbols and exponents, limited to zetta and below
prefixes = [
    ('zetta', 'Z', 21),
    ('exa', 'E', 18),
    ('peta', 'P', 15),
    ('tera', 'T', 12),
    ('giga', 'G', 9),
    ('mega', 'M', 6),
    ('kilo', 'k', 3),
    ('hecto', 'h', 2),
    ('deca', 'da', 1),
    (' - ', ' - ', 0),
    ('deci', 'd', -1),
    ('centi', 'c', -2),
    ('milli', 'm', -3),
    ('micro', r'$\mu$', -6),
    ('nano', 'n', -9),
    ('pico', 'p', -12),
    ('femto', 'f', -15),
    ('atto', 'a', -18),
    ('zepto', 'z', -21),
]

# Create DataFrame
df = pd.DataFrame(prefixes, columns=['Prefix', 'Symbol', 'Exp.'])

# Add columns for multiplier and notations
df['Mult.'] = df['Exp.'].apply(lambda x: r'$10^{%d}$' % x)
df['Scientific'] = df['Exp.'].apply(lambda x: r'$1 \times 10^{%d}$' % x)
df['E-notat.'] = df['Exp.'].apply(lambda x: '1e%d' % x if x >= 0 else '1e-%d' % abs(x))

# Create figure and axis
fig = plt.figure(figsize=(6.3, 6.3), facecolor='black')  # Set figure background to black
ax = fig.add_subplot(111)
ax.axis('off')
ax.set_facecolor('black')  # Set axis background to black

# Add title
plt.title('SI Prefixes and Number Notations', fontsize=14, fontweight='bold', color='white', pad=20)

# Create table
the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', 
                     cellColours=[['black']*len(df.columns)]*len(df),  # Black cell backgrounds
                     colColours=['black']*len(df.columns))  # Black header backgrounds

# Adjust table properties
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1.1, 1.4)  # Adjusted scaling for better fit with bold font

# Set bold font and white text for all cells
for key, cell in the_table.get_celld().items():
    cell.set_text_props(fontweight='bold', color='white')  # White text, bold
    cell.set_edgecolor('white')  # White cell borders

# Save to image
plt.savefig('numbers_table_bold_inverted.png', bbox_inches='tight', dpi=300, facecolor='black')
print("Table saved to 'numbers_table_bold_inverted.png'")