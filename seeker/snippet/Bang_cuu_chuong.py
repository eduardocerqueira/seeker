#date: 2021-10-21T16:54:38Z
#url: https://api.github.com/gists/f8653c32cb55e9b2ff629c13907aac12
#owner: https://api.github.com/users/hquynh2608

def input_data():
    while True:
        try:
            num = int(input())
            if num >= 1:
                return num
                break
            else:
                print("Nhập một số tự nhiên lớn hơn 1.: ")
                print()
        except ValueError:
            print("Bạn phải nhập một số nguyên ")
            print()

if __name__ == '__main__':
    print("Nhập số dòng: ")
    rows = input_data()
    print("Nhập số cột: ")
    columns = input_data()
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            print("{:5d}".format(column * row), end= " ")
        print()