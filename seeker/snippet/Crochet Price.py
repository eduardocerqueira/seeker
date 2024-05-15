#date: 2024-05-15T16:58:25Z
#url: https://api.github.com/gists/bb0d40cf45fd1b082ad8f7f265872566
#owner: https://api.github.com/users/asytrao

product = input("What is the product you want to make? ")
wool = int(input("What is the price of the wool? "))
price_per_hour = int(input("What is the price per hour? "))
hours = int(input("How many hours did you work? "))
other_costs = int(input("What are the other costs? "))

total_cost = wool + price_per_hour * hours + other_costs

print(f"The total cost of {product} is {total_cost}")