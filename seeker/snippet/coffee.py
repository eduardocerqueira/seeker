#date: 2024-01-01T16:44:30Z
#url: https://api.github.com/gists/6ab0cdfbc54da63b2080ca2858c0fa97
#owner: https://api.github.com/users/geeknik

def calculate_optimal_coffee():
    weight_kg = float(input("Enter your weight in kg: "))
    age = int(input("Enter your age: "))
    sensitivity = input("Enter your caffeine sensitivity (low/medium/high): ").lower()

    # Base caffeine limit
    caffeine_limit = 400  # FDA daily recommended limit in milligrams

    # Adjust caffeine limit based on age
    if age < 18 or age > 65:
        caffeine_limit *= 0.5

    # Adjust caffeine limit based on weight
    caffeine_limit *= weight_kg / 70  # 70 kg is the average adult weight

    # Adjust caffeine limit based on sensitivity
    if sensitivity == 'high':
        caffeine_limit *= 0.5
    elif sensitivity == 'low':
        caffeine_limit *= 1.5

    caffeine_per_cup = 95  # milligrams
    optimal_coffees = caffeine_limit / caffeine_per_cup
    return optimal_coffees

optimal_coffees = calculate_optimal_coffee()
print(f"Optimal number of coffees per day: {optimal_coffees:.2f}")