#date: 2025-01-01T16:43:24Z
#url: https://api.github.com/gists/5ec3bd334eaab5fddb2da3f88ca15f33
#owner: https://api.github.com/users/PieroPaialungaAI

y_pred = model.predict(X_test)
y_pred = [1 if y[0] > 0.5 else 0 for y in y_pred]
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
for i in range(1,11):
  plt.subplot(5,2,i)
  plt.title('Sequence n. %i'%(i))
  plt.plot(X_test[i])
  plt.title(f'Predicted class {y_pred[i]}, real class {int(y_test[i])}')
plt.tight_layout()