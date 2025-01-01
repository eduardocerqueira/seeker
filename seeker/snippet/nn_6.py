#date: 2025-01-01T16:41:19Z
#url: https://api.github.com/gists/0241d3da65da291ddd3c4f5388d78ffa
#owner: https://api.github.com/users/PieroPaialungaAI

sine_waves = build_sine_waves()
square_waves = build_square_waves()
all_waves = np.concatenate([sine_waves,square_waves])
labels = np.concatenate([np.zeros(len(sine_waves)),np.ones(len(square_waves))])

X = all_waves
y = labels
plt.plot(time_steps, X[0], label="Sine Wave")
plt.plot(time_steps, X[-1], label="Square Wave")
plt.legend()
plt.show()
random_integers = np.random.choice(len(X),size=len(X),replace=False)
X = X[random_integers]
y = y[random_integers]
X_train, y_train, X_test, y_test = X[:int(0.8*len(X))], y[:int(0.8*len(y))], X[int(0.8*len(X)):], y[int(0.8*len(y)):]
# Reshape X for the neural network (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Plot the sine and square wave
