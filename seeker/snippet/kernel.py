#date: 2025-02-11T16:56:30Z
#url: https://api.github.com/gists/d93d1cbd6b5a5bc37d0cb12b1c534c97
#owner: https://api.github.com/users/Zaryob

# 3. Morfolojik işlemler için çekirdek (kernel) tanımlayalım.
# Burada 5x5 boyutunda dikdörtgen bir çekirdek kullanıyoruz.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
