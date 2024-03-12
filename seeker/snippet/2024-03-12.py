#date: 2024-03-12T16:42:25Z
#url: https://api.github.com/gists/66ca68231bd520f4a10d1f3b2c982a39
#owner: https://api.github.com/users/codingexercisess

# Comma Code

def solution(values):
    # If the list is empty
    if not values:
        print('Error! Empty list was given.')
        return
    # If there's only one item in the list
    elif len(values) == 1:
        print(values[0] + '.')
        return
    # Joining all items except the last one with commas
    items = ', '.join(values[:-1])
    # Printing the formatted string with the last item preceded by "and"
    print(f"{items}, and {values[-1]}.")


# Example
planets = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn',
            'Uranus', 'Neptune', 'Pluto', 'Earth']
solution(planets)