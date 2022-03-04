#date: 2022-03-04T16:47:39Z
#url: https://api.github.com/gists/9950b36bacb453065d913674cb715907
#owner: https://api.github.com/users/a3darekar

## TO SAVE MODEL WEIGHTS TO FILE:
def save_local_weights(model):
	weights = selfmodel.get_weights()
	np.save(f'model_weights', weights) # Saved model weights will be in 'model_weights.npy'
	# np.save('weights', weights)
	print("Saved Local weights")