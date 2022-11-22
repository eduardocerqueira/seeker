#date: 2022-11-22T17:10:40Z
#url: https://api.github.com/gists/204a70e2a69042adb3a13d1e6edeffd5
#owner: https://api.github.com/users/gabri-al

# Create the new dataset to be plotted: [centroid_1, centroid_0, the 12 test set points]
data_ = [centroid_1, centroid_0]
for i in np.arange(0, len(X_test), 1):
    data_.append(X_test[i])
    
# Plot with PCA
pca = PCA()
p1 = pca.fit_transform(np.array(data_))
cen_1 = p1[0]
cen_2 = p1[1]
test_1 = p1[2:8]
test_2 = p1[8:]
plt.scatter(cen_1[0],cen_1[1], c='darkblue', label='1-Metal Centroid', marker = "D")
plt.scatter(cen_2[0],cen_2[1], c='crimson', label='0-Pop Centroid', marker = "D")
plt.scatter(test_1[:,0],test_1[:,1], c='steelblue', label='1-Metal Test', marker = "+")
plt.scatter(test_2[:,0],test_2[:,1], c='palevioletred', label='0-Pop Test', marker = "+")
plt.title("Test Set with Playlist centroids | PCA")
plt.legend(loc='lower right', prop={'size': 6})
plt.rcParams['figure.dpi'] = 100
plt.show()