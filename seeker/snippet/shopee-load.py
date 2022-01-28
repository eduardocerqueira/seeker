#date: 2022-01-28T16:54:34Z
#url: https://api.github.com/gists/a5ffb66a1af94d2524d1389b902e434b
#owner: https://api.github.com/users/vkhangpham

from datasets import load_dataset

raw_data = load_dataset('csv', data_files='data/shopee_train.csv')
raw_data = raw_data['train'].train_test_split(test_size=.1)
raw_train = raw_data['train']
raw_test = raw_data['test']