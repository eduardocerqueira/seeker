#date: 2022-01-11T17:04:33Z
#url: https://api.github.com/gists/0f7d1befc192557e77f3c0e33f271fae
#owner: https://api.github.com/users/Yash22222

score = float(input("Enter Your Score"))
if (score >=100):
   print("Wrong Input")
elif score>=0.9:
   print("A")
elif score>=0.8:
   print("B")
elif score>=0.7:
   print("C")
elif score>=0.6:
   print("D")
else:
   print("F")