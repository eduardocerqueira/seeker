#date: 2021-12-17T16:55:14Z
#url: https://api.github.com/gists/c6d78b8dd0dca72f605046c2d017d97e
#owner: https://api.github.com/users/yuyueugene84

from random import randint

print("-------歡迎來到剪刀石頭布！-------")
name = input("請輸入您的名稱：")

while True:
    # 記錄玩家與電腦勝利次數
    winners = {
        "player": 0,
        "computer": 0,
        "draw": 0
    }
    
    gestures = {
        1: "剪刀",
        2: "石頭",
        3: "布"
    }
    # 將猜拳流程重複3次
    for i in range(1, 4):
        while True:
            user_hand = input("請出拳 (1) 剪刀 (2) 石頭 (3) 布：")
            # 若玩家輸入的内容是數字，而且是 1, 2, 3 的其中一個
            if user_hand.isdigit() and (user_hand in ["1", "2", "3"]):
                # 跳出 loop
                user_hand = int(user_hand)
                break
            else:
                print("請輸入合法的選項！")

        comp_hand = randint(1,3)

        print("==================================")
        print(f"{name} 出了: {gestures[user_hand]}, 電腦出了: {gestures[comp_hand]}")
        print("==================================")

        if user_hand == comp_hand:
            print("平手!")
            winners["draw"] += 1
        elif user_hand == 1 and comp_hand == 3:
            print("你贏了一把!")
            winners["player"] += 1
        elif user_hand == 2 and comp_hand == 1:
            print("你贏了一把")
            winners["player"] += 1
        elif user_hand == 3 and comp_hand == 2:
            print("你贏了一把")
            winners["player"] += 1
        else:
            print("你輸了一把!")
            winners["computer"] += 1
        # 每次猜完拳。就顯示積分
        print(f"===================================")
        print(f"|{name.center(16)}|{'電腦'.center(16)}|")
        print(f"===================================")
        print(f"|{str(winners['player']).center(16)}|{str(winners['computer']).center(16)}|")
        print(f"===================================")
    # 玩過3局
    print("==================================")
    print(f"{name}贏了: {winners['player']} 把")
    print(f"電腦贏了: {winners['computer']} 把")
    print(f"平手: {winners['draw']} 把")
    print("==================================")
    # 判斷最終勝利者
    if winners["player"] > winners["computer"]:
        print("恭喜你是最終勝利者！")
    elif winners["player"] < winners["computer"]:
        print("哭哭，你輸了！")
    else:
        print("平手！")
    # 詢問玩家是否再玩一次
    while True:
        keep_play = input("想要再玩一次嗎？ Y) 是 N) 否：")
        # 若玩家輸入的内容是數字，而且是 1, 2, 3 的其中一個
        if keep_play in ["Y", "y", "N", "n"]:
            # 跳出 loop
            break
        else:
            print("請輸入合法的選項！")
    
    if keep_play.upper() == "N":
        break

print("感謝你玩剪刀石頭布！")