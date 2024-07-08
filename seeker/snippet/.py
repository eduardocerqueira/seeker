#date: 2024-07-08T16:52:23Z
#url: https://api.github.com/gists/473478b166cdd7814c9c5a3fdfe5660e
#owner: https://api.github.com/users/webThorstenTest

import json
from typing import List
import time

class HitTestable:

    def testHit(self, x,y) -> bool:
        raise NotImplementedError()

class I(HitTestable):

    width = 0
    height = 0
    top_left_corner = (0,0)

    def __init__(self, width, height, top_left_corner) -> None:
        self.width = width
        self.height = height
        self.top_left_corner = top_left_corner 
        super().__init__()
    def testHit(self, x, y) -> bool:
        x = x - self.top_left_corner[0]
        y = y - self.top_left_corner[1]
        return x >= 0 and x <= self.width and y >= 0 and y <= self.height
             

class O(HitTestable):

    width = 0
    radius = 0
    center = (0, 0)

    def __init__(self, width, diameter, center) -> None:
        self.width = width
        self.radius = diameter/2
        self.center = center
        super().__init__()

    def testHit(self, x, y) -> bool:
        x = x - self.center[0]
        y = y - self.center[1]
        diff_from_center = (x**2+y**2)**(1/2)
        return diff_from_center <= self.radius and diff_from_center > self.radius - self.width 



dataset = {}
with open("dataset.json", "r") as file:
    dataset = json.load(file)

    letters: List[HitTestable] = [I(20, 150, (145, 75)), O(20, 150, (250, 150)), O(20, 150, (410, 150))]
    hit_counter = 0
    points = dataset["coords"]

    x_lower_bound = 145
    x_upper_bound = 485
    y_lower_bound = 75
    y_upper_bound = 225
    bounds = ((145,75), (485, 225))

    for point in points:
        x, y = point
        if(x< x_lower_bound or x > x_upper_bound or y < y_lower_bound or y > y_upper_bound):
            continue
        if any(letter.testHit(x, y) for letter in letters):
            hit_counter += 1

    print(f"Hits: {hit_counter}")