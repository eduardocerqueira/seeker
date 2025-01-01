#date: 2025-01-01T16:38:50Z
#url: https://api.github.com/gists/8561dadbea0c992bd7bf1eec347db28f
#owner: https://api.github.com/users/PieroPaialungaAI

# Predict the next value
predicted = model.predict(X_test)
t_X_test= np.array(t_X)[ind_order[int(0.8*len(X)):]]
t_Y_test = np.array(t_Y)[ind_order[int(0.8*len(X)):]]
# Plot the results
plt.figure(figsize=(10,10))
for i in range(1,10):
  plt.subplot(3,3,i)
  j = np.random.randint(0,len(X_test))
  plt.title('Sequence n. %i'%(j))
  plt.plot(t_X_test[j],X_test[j],marker='x',label='Current sequence')
  plt.plot(t_Y_test[j],y_test[j],'x',label='Next point')
  plt.plot(t_Y_test[j],predicted[j][0],label='Predicted',marker='x')
  plt.xlabel('Time (t)')
  plt.ylabel('Amplitude')
  plt.legend(loc='best',fontsize=8)
plt.tight_layout()