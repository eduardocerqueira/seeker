#date: 2025-02-11T16:59:12Z
#url: https://api.github.com/gists/2ee859415aa498bf932794c880d4d51d
#owner: https://api.github.com/users/Zaryob

# c) Opening (Açma): Erosion sonrası dilation (küçük gürültülerin giderilmesi için).
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# d) Closing (Kapama): Dilation sonrası erosion (küçük deliklerin doldurulması için).
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)