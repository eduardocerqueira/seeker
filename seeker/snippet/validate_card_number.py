#date: 2024-01-09T16:49:49Z
#url: https://api.github.com/gists/8818a46aa05ed97c8e0685d55ff79969
#owner: https://api.github.com/users/hashem-nowruzi

card_number = input("Card Number: ")
result = sum(int(num) for num in "".join(str(int(number) * 2) if index % 2 == 0 else number for index, number in enumerate(card_number))) % 10 == 0
print(result)