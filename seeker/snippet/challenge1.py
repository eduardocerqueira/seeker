#date: 2023-08-03T16:55:11Z
#url: https://api.github.com/gists/fb20f376a676a4776efbabe67e42258e
#owner: https://api.github.com/users/greatvijay

#Challenge 1
class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def sqSum(self):
        return self.x ** 2 + self.y ** 2 + self.z ** 2

# Testing the Point class
point = Point(1, 3, 5)
print(point.sqSum())  
