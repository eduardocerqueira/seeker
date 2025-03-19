#date: 2025-03-19T17:08:19Z
#url: https://api.github.com/gists/8dbd1b6d8cc56e5a3bd255ba8c5aa016
#owner: https://api.github.com/users/ilkermanap

# Tekrarlayan sayiyi bul. len(nums) + 1 boyunda dizi kullan,
# yeni dizi kullanma. sadece dizinin sonundaki bos yeri kullanarak coz

nums1 = [1,2,6,4,5,3,5]
nums2 = [1,3,4,2,2]
nums3 = [3,1,3,4,2]

nums4 = [3,1,2,1]

def repeating(dizi):
    i = 0
    dizi = dizi +[0]
    # 3,1,2,1,0

    print(dizi)
    for sayi in dizi[:-1]:
        dizi[-1] = sayi
        3,1,2,1,3
        i += 1
        for yenisayi in dizi[i:-1]:
            if yenisayi == dizi[-1]:
                return yenisayi


if __name__ == "__main__":
   print(repeating(nums1))
   print(repeating(nums2))
   print(repeating(nums3))
    
        
    
