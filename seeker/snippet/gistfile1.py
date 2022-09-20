#date: 2022-09-20T17:02:51Z
#url: https://api.github.com/gists/cc32276c1364afeb3f582499f6995048
#owner: https://api.github.com/users/poloniki

def rev_sigmoid(x):
    return (1 / (1 + math.exp(0.5*x)))
    
def activate_similarities(similarities, p_size=10):
  # Create reversed sigmoid activation

  x = np.linspace(-10,10,p_size)
  y = np.vectorize(rev_sigmoid)
  y = np.pad(y(x),(0,similarities.shape[0]-p_size))

  # Get every diagonal in our similarity matrix
  diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
  # Because each diagonal is different length we should pad it with zeros at the end
  diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
  # Lets combine the result into matrix
  diagonals = np.stack(diagonals)
  # Multiply similarities with our activation
  diagonals = diagonals * y.reshape(-1,1)
  # Get the sum of activated similarities
  activated_similarities = np.sum(diagonals, axis=0)
  return activated_similarities
  
  
change_points = activate_similarities(similarities, p_size=5)

fig, ax = plt.subplots()
# for all local minimals
minmimals = argrelextrema(change_points, np.less, order=2)
sns.lineplot(y=median_similarity, x=range(len(median_similarity)), ax=ax).set_title('Relative minima');
# Now lets plot vertical line in order to understand if we have done what we wanted
plt.vlines(x=minmimals, ymin=min(median_similarity), ymax=max(median_similarity), colors='purple', ls='--', lw=1, label='vline_multiple - full height')
