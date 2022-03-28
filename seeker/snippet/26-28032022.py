#date: 2022-03-28T17:14:37Z
#url: https://api.github.com/gists/de2f20e94e2233a9a4df75cfda29e8d6
#owner: https://api.github.com/users/Lectorem794643

# захотел вам показать
# из читов только переводчик для читаемости программы

with open('inf_22_10_20_26.txt', 'r') as f:
    new_data = [int(x) for x in f]
N = new_data[0]
cost = sorted(new_data[1::])
save_cost = sorted([x for x in cost if x <= 100]) # len == 94
lose_cost = sorted([x for x in cost if x > 100])  # len == 906
cheque = list()
for I in save_cost:
    cheque.append(I)
    cheque.append(max(lose_cost))
    lose_cost.remove(max(lose_cost))
# на данном этипе мы разложили и однулили самые пригрышные покупки, а так же исчерпали дешевые товары и самые дорогие
# далее будем раскладывать то что осталось по принципу мах - мин
while len(lose_cost) > 0:
    cheque.append(min(lose_cost))
    lose_cost.remove(min(lose_cost))
    cheque.append(max(lose_cost))
    lose_cost.remove(max(lose_cost))
print(cheque)
print(len(cheque))
# все разложили ничего не потеряли :D
final_cost = 0
checksum = 0
the_most_expensive_product = 0
for I in range(len(cheque)):
    if I % 2 == 0 and cheque[I] > 100:
        final_cost += int(cheque[I] * 0.7) + 1
        checksum += 1
        the_most_expensive_product = max(the_most_expensive_product, cheque[I])
    elif I % 2 == 0 and cheque[I] <= 100:
        final_cost += cheque[I]
        checksum += 1
    elif I % 2 != 0:
        final_cost += cheque[I]
        checksum += 1
if checksum == N:
    print('OK')
print(final_cost, the_most_expensive_product)