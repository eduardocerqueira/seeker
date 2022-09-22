#date: 2022-09-22T17:14:56Z
#url: https://api.github.com/gists/f247992837f2b298466924edf42af0a8
#owner: https://api.github.com/users/Suleyalim

num1 = int(input("İlk sayiyi giriniz: "))
num2 = int(input("İkinci sayiyi giriniz: "))

if num1 >= num2:
  if num1 == num2:
    print("{} değeri {} değerine eşittir".format(num1,num2))
  else:
    print(f"{num1} değeri {num2} değerinden büyüktür")
else:
  print(f"{num1} değeri {num2} değerinden küçüktür")