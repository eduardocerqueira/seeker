#date: 2022-02-23T17:11:04Z
#url: https://api.github.com/gists/9a4c048e1850e1d2f197a9e2fc077e13
#owner: https://api.github.com/users/ecdedios

# Performing OTSU threshold
ret, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
plt.figure(figsize=(25, 15))
plt.imshow(threshold_image, cmap='gray')
plt.show()