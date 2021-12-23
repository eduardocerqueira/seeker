#date: 2021-12-23T16:38:45Z
#url: https://api.github.com/gists/1582d5fe74b6caf3e76d0eaa0087310c
#owner: https://api.github.com/users/wazir19gh

from sklearn.decomposition import PCA

n_component_list = [5,10,20,30,40]

for n_c in n_component_list:
  pca = PCA(n_components=n_c)
  pca.fit(vgg16_df.iloc[:,:-1])
  print("The cummulative variance explained by the top {} components is {}".format(n_c,sum(pca.explained_variance_ratio_)))