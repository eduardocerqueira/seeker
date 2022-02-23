#date: 2022-02-23T17:08:55Z
#url: https://api.github.com/gists/4036c5441400d3f40e3c37afa4b35f91
#owner: https://api.github.com/users/ecdedios

# convert the image to grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(25, 15))
plt.imshow(gray_image, cmap='gray')
plt.show()