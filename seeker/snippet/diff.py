#date: 2022-07-11T16:59:38Z
#url: https://api.github.com/gists/95aadf106fc922f67b240560af670917
#owner: https://api.github.com/users/bergio13

difference = dca - lump_sum
difference.plot(figsize=(15, 9), label='Difference', lw=0.5);
plt.fill_between(difference.index, y1=difference, y2=0, color='green', where=difference>0, label='DCA > Lump Sum');
plt.fill_between(difference.index, y1=difference, y2=0, color='red', where=difference<0, label='DCA < Lump Sum');
plt.title('Difference: DCA - Lump Sum');
plt.legend();