#date: 2024-01-17T16:45:54Z
#url: https://api.github.com/gists/6a8080b54758901fd428a09a9faafe0d
#owner: https://api.github.com/users/pratapsdev11

def standardize_data(x):
  std_data=x-mean(x)/std(x)
  return std_data