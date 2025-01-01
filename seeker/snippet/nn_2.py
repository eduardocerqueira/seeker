#date: 2025-01-01T16:36:34Z
#url: https://api.github.com/gists/6ce10c3901cbb8a6309b6e0972a52b51
#owner: https://api.github.com/users/PieroPaialungaAI

plt.figure(figsize=(10,10))
for i in range(1,10):
  plt.subplot(3,3,i)
  j = np.random.randint(0,len(X))
  plt.title('Sequence n. %i'%(j))
  plt.plot(t_X[j],X[j],marker='x',label='Current sequence')
  plt.plot(t_Y[j],y[j],'x',label='Next point')
  plt.xlabel('Time (t)')
  plt.ylabel('Amplitude')
  plt.legend(loc='best',fontsize=8)
plt.tight_layout()