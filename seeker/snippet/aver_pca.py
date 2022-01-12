#date: 2022-01-12T17:05:53Z
#url: https://api.github.com/gists/b8ed4cd1abdb94d990423730c90e7cca
#owner: https://api.github.com/users/haykaza

# get average length of sentences for finding the optimal number of PCA components
length = []
for i in range(len(input_df)):
    length.append(len(input_df['text'][i]))
print(f'Average length of our text input is {round(np.mean(length),0)}')