#date: 2023-12-04T16:51:24Z
#url: https://api.github.com/gists/3712be8625cedfca8c21295ce7a7d20f
#owner: https://api.github.com/users/amiraday

"""
כתבו תוכנית כספומט בלולאת while
אשר מבקשת שם וסיסמה רק במידה ושתי הפרטים נכונים יקבל תפריט במידה ולא יקבל הודעה
שם או סיסמה לא נכונים  ושוב התוכנית תחזור לבקש שם וסיסמה . יש לו 3 ניסיונות בכל פעם יודפס מספר הניסיונות הנותרים אם אפשרות ליציאה אחרי 3 ניסיונות יודפס לו לגשת לסניף לקבלת הכרטיס
1 בירור יתרה 2 משיכה 3 הפקדה
ללקוח 1000 שקלים בחשבון
במידה ויבחר 1 יופיע לו היתרה. נשאל אם רוצה לצאת או להמשיך יחזור לתפריט
במידה ויבחר 2 יוכל למשוך עד 1000 שקלים מה שיש בחשבון, אחרי המשיכה תוצג לו היתרה נשאל אם רוצה לצאת או להמשיך יחזור לתפריט
במידה ויבחר 3 יפקיד כסף ותוצג לו היתרה כולל סכום ההפקדה נשאל אם רוצה לצאת או להמשיך יחזור לתפריט
במידה ויבחר 4 יציאה
בסיום נשאל את המשתמש אם ברצנו לצאת אם כן יודפס שלום אלי
אם לא, יחזור לתפריט
"""
stop = "no"
money_in_the_bank = 1000
max_attempts = 3
counter = 0
while stop == "no" :
    name = input("Enter a username ")
    password = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"a "**********"m "**********"e "**********"  "**********"= "**********"= "**********"  "**********"" "**********"a "**********"m "**********"i "**********"r "**********"" "**********"  "**********"a "**********"n "**********"d "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"= "**********"= "**********"  "**********"" "**********"1 "**********"2 "**********"3 "**********"4 "**********"5 "**********"" "**********"  "**********": "**********"
        print("You've logged in successfully ")
        while True :
              menu = input(f"hello  {name}  What do you want to do?      \n 1.Balance \n 2.withdrawal \n 3.Deposit \n 4.Exit  \n ")
              if menu == "1" : print(f"Your balance is:  {money_in_the_bank}")
              elif menu == "2" :
                  draw = int(input(f"Your balance is:  {money_in_the_bank} How much do you want to withdraw?"))
                  if draw > money_in_the_bank : print("you do not have enough money")
                  else: print(f"Continued successfully {draw}   Your current account balance {money_in_the_bank - draw} ")
              elif menu == "3" :
                  Deposit = int(input(f"Your balance is:  {money_in_the_bank }  How much would you like to deposit?" ))
                  print(f"Your new balance    {money_in_the_bank + Deposit}")
              elif menu =="4" : print(f"Thank you {name} have a good day") ; exit()
              else: print("Try again wrong choice 1-4")
    else:
        counter += 1
        print(f"Incorrect password or username , You have {max_attempts - counter}attempts")
        if counter >= max_attempts : print("The card has been swallowed. Please come to your branch to receive a new card") ; exit()




