#date: 2022-12-13T16:49:00Z
#url: https://api.github.com/gists/54338ea3932cb434e9e775199d1effad
#owner: https://api.github.com/users/MishaelRosenthal

from fractions import Fraction as frac
from collections import defaultdict

class JeepProblem:
    
    def __init__(self, n):
        self.location = frac(0)
        self.fuel_in_tank = frac(1)
        self.n = n
        self.fuel_dumps = defaultdict(lambda: frac(0))
        self.fuel_dumps[0] = frac(self.n-1)
        self.farthest = frac(0)

    def dump_fuel(self, location, quantity):
        self.fuel_dumps[location] += quantity
        self.fuel_in_tank -= quantity
    
    def pick_up_fuel(self,location, quantity):
        self.fuel_dumps[location] -= quantity
        self.fuel_in_tank += quantity
        
    def drive(self, delta):
        self.fuel_in_tank -= abs(delta)
        self.location += delta
        self.farthest = max(self.farthest, self.location)
        
    def print_status(self):
        print("location", self.location)
        print("fuel_in_tank", self.fuel_in_tank)
        print("fuel_dumps[location]", self.fuel_dumps[self.location])
        print()

    def run(self):
        self.print_status()
        
        for k in range(1, self.n+1):
            for j in range(self.n, self.n-k, -1): #reversed(range(self.n+1-k, self.n+1)):
                delta = frac(frac(1, 2*j))
                self.drive(delta)
                
                if(j==self.n+1-k):
                    self.dump_fuel(self.location, frac(self.n-k, self.n-k+1))
                else:
                    self.pick_up_fuel(self.location, delta)
                
                self.print_status()
                    
            for j in range(self.n+1-k, self.n+1):
                delta = frac(frac(1, 2*j))
                self.drive(-delta)
               
                if(j!=self.n):
                    self.pick_up_fuel(self.location, frac(1, 2*(j+1)))
                elif(k!=self.n):
                    self.pick_up_fuel(self.location, frac(1))
                
                self.print_status()
                
        print("farthest we got from base was", self.farthest)
        print("Done!!!")
            
if __name__ == '__main__':
    input_str = input("enter a whole small number")
    if input_str.isnumeric():
        n = int(input_str)
        JeepProblem(n).run()
    else:
        print("The string", input_str, "isn't a number")
   