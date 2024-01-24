#date: 2024-01-24T17:08:15Z
#url: https://api.github.com/gists/5890ce02e704d64e29d1fdfd886a73fa
#owner: https://api.github.com/users/YuriiYatcynych

apple_price_per_kg = float(input("Введіть ціну за 1 кг яблук: "))
pear_price_per_kg = float(input("Введіть ціну за 1 кг груш: "))
apple_weight = float(input("Введіть вагу яблук: "))
pear_weight = float(input("Введіть вагу груш: "))

apple_cost = apple_price_per_kg * apple_weight
pear_cost = pear_price_per_kg * pear_weight

total_cost = apple_cost + pear_cost

print("Вартість яблук =", apple_cost)
print("Вартість груш =", pear_cost)
print("Загальна вартість =", total_cost)