#date: 2023-01-10T17:01:27Z
#url: https://api.github.com/gists/1dedfadd7794838e443901ff80adfc78
#owner: https://api.github.com/users/catslovedata

from matplotlib import pyplot as plt
import matplotlib as mpl
import math

target = 5000 # Steps per day target

plt.figure(figsize=(6,6))

# Used in calculating axes
# Assumes a zero base. If the base should be something different then adjust accordingly
max_data_value = max(single_month_data)
y_min = max_data_value * -0.75
y_max = max_data_value * 1.25
outside_ring_height = max_data_value * 0.2
outside_ring_base = max_data_value * 1.1

segment_size = math.radians(360 / len(single_month_data))
x_positions = [ind * segment_size for ind, _ in enumerate(single_month_data)]

# cmap = mpl.cm.get_cmap('RdYlGn')
cmap=custom_cmap

ax = plt.subplot(1, 1, 1, polar=True);
ax.grid(False)
ax.axis(False)
ax.set_theta_direction(-1)
ax.set_theta_offset(math.radians(90))
# Adjustment to accommodate the centre and outside ring. Change according to the desired proportion.
ax.set_ylim([y_min, y_max])
ax.set_title('Steps Count', fontsize=20, fontname='Open Sans', fontweight='300', color='#222222', alpha=0.9)

# Plot outside bar for KPI
kpi = min(target, sum(single_month_data)/len(single_month_data))
kpi_proportion = kpi / target   # What about floating point errors?
kpi_achieved_width = math.radians((kpi_proportion) * 360)
kpi_remaining_width = math.radians(360) - kpi_achieved_width
kpi_color = cmap(kpi_proportion)

ax.bar(x=0, height=outside_ring_height, bottom=outside_ring_base, width=kpi_achieved_width, color=kpi_color, alpha=1, align='edge')
if kpi < target:
    ax.bar(x=kpi_achieved_width, height=outside_ring_height, bottom=outside_ring_base, width=kpi_remaining_width, color='#cccccc', alpha=0.3, align='edge')

# Plot the target
ax.bar(x=0, height=target,bottom=0, width=math.radians(360), color='#cccccc', alpha=0.2,)

# Plot data bars & labels for the days
data_colors = [cmap(min(x, target) / target) for x in single_month_data]

ax.bar(x_positions, single_month_data, width = segment_size * 0.9, align='edge', color=data_colors, alpha=0.7)
for ind, x in enumerate(x_positions):
    ax.text(x + segment_size/2,max_data_value * -0.1,str(ind+1), ha='center', va='center', fontsize=6, fontname='Open Sans', fontweight='300', color='#222222', alpha=0.9)

# Write center text
ax.text(0, y_min, '{0} total\n{1} average'.format(sum(single_month_data), int(round(sum(single_month_data) / len(single_month_data), 0))), ha='center', va='center', fontsize=10, fontname='Open Sans', fontweight='300', color='#222222')
