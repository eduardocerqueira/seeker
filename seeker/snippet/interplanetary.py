#date: 2025-01-14T17:01:16Z
#url: https://api.github.com/gists/ef76bf041746bbc5b21432a54457d47b
#owner: https://api.github.com/users/nikoheikkila

Weights = dict[str, float]

def calculate_planetary_weight(weight_kg: float) -> Weights:
    """
    Calculate a person's weight on different planets in the Solar System.
    Input weight should be in kilograms.
    Returns a dictionary with weights on each planet in kilograms.
    """

    # Dictionary of relative surface gravities compared to Earth
    # These values represent how strong each planet's gravity is compared to Earth's
    # For example, on Mercury you would weigh only 37.8% of your Earth weight
    relative_gravities = {
        'Mercury': 0.378,
        'Venus': 0.907,
        'Earth': 1.000,
        'Mars': 0.377,
        'Jupiter': 2.528,
        'Saturn': 1.065,
        'Uranus': 0.886,
        'Neptune': 1.137
    }

    # Calculate weight on each planet
    # Weight on a planet = Earth weight * planet's relative gravity
    # We round to 2 decimal places for cleaner output
    decimals = 2
    planetary_weights = {}
    for planet, gravity in relative_gravities.items():
        weight_on_planet = weight_kg * gravity
        planetary_weights[planet] = round(weight_on_planet, decimals)

    return planetary_weights

def display_weights(weights: Weights) -> None:
    """
    Display the calculated weights in a formatted manner.
    Uses kilograms and includes proper spacing for readability.
    """
    print("\nYour weight on different planets:")
    print("–" * 40)
    print(f"{'Planet':<12} {'Weight (kg)':>15}")
    print("–" * 40)

    for planet, weight in weights.items():
        print(f"{planet:<12} {weight:>15,.2f}")

def main() -> None:
    try:
        # Get weight input from user in kilograms
        weight = float(input("Enter your weight in kilograms: "))

        # Validate input to ensure weight is positive
        # This helps prevent unrealistic calculations
        if weight <= 0:
            print("Weight must be a positive number!")
            return

        # Calculate weights and display results
        planetary_weights = calculate_planetary_weight(weight)
        display_weights(planetary_weights)

    except ValueError:
        print("Please enter a valid number for weight!")

if __name__ == "__main__":
    main()
