#date: 2022-04-15T16:48:15Z
#url: https://api.github.com/gists/530026e1806c571dfccaabf701c33bef
#owner: https://api.github.com/users/ShubhashreeSur

test_data_len = X_test.shape[0]

# we create a output array that has exactly same size as the CV data
test_predicted_y = np.zeros((test_data_len,12))
for i in range(test_data_len):
    rand_probs = np.random.rand(1,12)
    test_predicted_y[i] = rand_probs/rand_probs[0].sum()
print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))