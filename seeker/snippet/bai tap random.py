#date: 2025-08-27T17:10:48Z
#url: https://api.github.com/gists/3a4e857d605901beeefb0ceb0adc08a1
#owner: https://api.github.com/users/thaopham31241027486-coder

#1.Guess The Number
import random
money = 100
win = 0
lose = 0

print("1. Guess The Number")
while money >= 5:
        print(f"Bạn hiện có {money}$")
        while True:
            level = input("Chọn độ khó 10_easy/6_medium/3_hard : ").lower()
            if level == "10":
                attempts = 10
                break
            elif level == "6":
                attempts = 6
                break
            elif level == "3":
                attempts = 3
                break
            else:
                print("Chọn sai, vui lòng nhập lại!")

        money -= 5

        secret_number = "**********"
        print("Nhập số bạn đoán: ")

        for i in range(1, attempts + 1):
            guess = int(input(f"Lần đoán thứ {i}/{attempts}: "))
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"g "**********"u "**********"e "**********"s "**********"s "**********"  "**********"= "**********"= "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"n "**********"u "**********"m "**********"b "**********"e "**********"r "**********": "**********"
                print("YOU WIN!")
                win += 1
                break
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"g "**********"u "**********"e "**********"s "**********"s "**********"  "**********"< "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"n "**********"u "**********"m "**********"b "**********"e "**********"r "**********": "**********"
                print("Số đúng lớn hơn")
            else:
                print("Số đúng nhỏ hơn")
        else:
            print(f"YOU LOSE! Số đúng là {secret_number}")
            lose += 1

        choice = input("Bạn có muốn chơi tiếp không? (y/n): ").lower()
        if choice != 'y':
            break

print("KẾT QUẢ CỦA BẠN: ")
print(f"WIN: {win}, LOSE: {lose}")
print(f"Số tiền còn lại: {money}$")
print("END GAME")

#2. Tài Xỉu
import random
money = 100
win = 0
lose = 0

print("2. GAME TÀI XỈU ")
while money > 0:
        print(f"Bạn hiện có {money}$")
        bet = int(input("Bạn muốn cược bao nhiêu? "))
        if bet > money or bet <= 0:
            print("Số tiền cược không hợp lệ!")
            continue

        choice = input("Bạn đoán Tài hay Xỉu? (tai/xiu): ").lower()

        xx1 = random.randint(1, 6)
        xx2 = random.randint(1, 6)
        tong = xx1 + xx2

        print(f"Kết quả: {tong}")
        if tong > 5 :
            print('=> TÀI')
        else:
            print('=> XỈU')

        result = "tai" if tong > 5 else "xiu"

        if choice == result:
            print("THẮNG CƯỢC!")
            money += bet
            win += 1
        else:
            print("THUA CƯỢC!")
            money -= bet
            lose += 1

        if money <= 0:
            print("Bạn đã hết tiền!")
            break

        ctn = input("Bạn có muốn chơi tiếp không? yes/no: ").lower()
        if ctn != "yes":
            break


print("KẾT QUẢ CỦA BẠN")
print(f"WIN: {win}, LOSE: {lose}")
print(f"Số tiền còn lại: {money}$")
print("END GAME")

