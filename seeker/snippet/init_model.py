#date: 2022-03-04T16:47:39Z
#url: https://api.github.com/gists/9950b36bacb453065d913674cb715907
#owner: https://api.github.com/users/a3darekar

def init_model(model_weights=None):
	# define the keras model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	if model_weights:
		if isinstance(model_weights, list):
			model.set_weights(model_weights)
		elif os.path.isfile(model_weights):
			print("Loading model with saved weights")
			weights = np.load(model_weights, allow_pickle=True)
			model.set_weights(weights)
		else:
			print("Unrecognizable weights type: ", type(model_weights))
			pass
	else:
		print("Loading fresh model")
		
	return model.compile(
		loss=tensorflow.keras.losses.categorical_crossentropy,
		optimizer=tensorflow.keras.optimizers.Adadelta(),
		metrics=['accuracy']
		)