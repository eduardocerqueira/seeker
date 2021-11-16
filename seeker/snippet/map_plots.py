#date: 2021-11-16T16:54:22Z
#url: https://api.github.com/gists/e2a2e0d501667f2b1da082f81c209346
#owner: https://api.github.com/users/Rosscoperry

try:
  

  filenames = []

  vmin1 = 0
  vmax1 = sensors_p1.select_dtypes(include=[np.number]).max().max()
  vmin2 = 0
  vmax2 = sensors_p2.select_dtypes(include=[np.number]).max().max()

  #assume this goes from start num to end numb
  for (i,j) in tqdm(zip(range(start_before,end_before),range(start_after,end_after))):
      fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(20, 12))
      #print(i)
      #print(j)
    #before
      # plot the line chart
      p1 = merged.columns.values[i]
      p2 = merged2.columns.values[i]
      plot_map3(merged,merged2, p1, p2, vmin1, vmax1, vmin2, vmax2, ax1, ax2, fig)

      p1 = merged.columns.values[j]
      before = datetime.strptime(p1, '%Y-%m-%d %H:%M:%S') #format 2020-08-17 00:00:00
      after = datetime.strftime(before, "%d/%B/%Y")

      p2 = merged2.columns.values[j]

      plot_map3(merged,merged2, p1, p2, vmin1, vmax1, vmin2, vmax2, ax3, ax4, fig)

      plt.text(-2.52, 57.26, "AFTER LOCKDOWN", fontsize=16, ha='center',
              va='top', wrap=True)
      plt.text(-2.52, 57.48, "BEFORE LOCKDOWN", fontsize=16, ha='center',
              va='top', wrap=True)
      plt.text(-2.52, 57.50, after, fontsize=20, ha='center',
              va='top', wrap=True)
      
      # create file name and append it to a list
      filename = f'{i}.png'
      filenames.append(filename)
    
      # save frame
      plt.savefig(filename)
      plt.close()

  # build gif
  with imageio.get_writer('/content/maps_before_after.mp4', mode='I') as writer:
      for filename in filenames:
          image = imageio.imread(filename)
          writer.append_data(image)
        
  # Remove files
  for filename in set(filenames):
      os.remove(filename)

except:
  for filename in set(filenames):
        os.remove(filename)