#date: 2021-10-21T17:05:02Z
#url: https://api.github.com/gists/0923c6185830de0f9def1fa9eade068f
#owner: https://api.github.com/users/hquynh2608

def input_string():
    string = input("Hãy viết một câu: ")
    return string

def count_char(string):
    count = 0
    for i in string:
        count += 1
    return count

string = input_string()
print(f"Số kí tự là: {count_char(string)}")