#date: 2023-10-27T16:52:59Z
#url: https://api.github.com/gists/40057e6254e8c9f36f5aa82d3302aa9f
#owner: https://api.github.com/users/Sarves21

# Body Mass Index Calculator
height = input()
weight = input()
height = float(height)
weight = int(weight)
BMI = weight / (height**2)
print(int(BMI))