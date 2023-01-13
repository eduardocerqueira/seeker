#date: 2023-01-13T16:40:12Z
#url: https://api.github.com/gists/70bdb3180b1faf07c4860a4127358071
#owner: https://api.github.com/users/jmccardle

# How many bananas can you bring 1,000 miles away, if your camel eats 1 banana per mile?

class CamelJourney:
    def __init__(self, distance=1000, start=3000, capacity=1000):
        self.map = [start] + [0] * (distance - 1)
        self.aboard = 0
        self.position = 0
        self.capacity = capacity
        self.current_cache = 0
        self.N = 4 # tuned for this puzzle
    
    def can_graze(self):
        return self.map[self.position] > 0
    
    def can_sow(self):
        return self.aboard > 0
    
    def graze(self, direction = 1):
        #ssert self.map[self.position] > 0
        self.map[self.position] -= 1
        self.position += direction
    
    def sow(self):
        #assert self.aboard > 0
        self.aboard -= self.N + 1 # eat & drop
        self.map[self.position] += self.N # dropped
        self.position += 1
        
    def eat(self):
        self.aboard -= 1
        self.position += 1
        
    def carry_forward(self):
        pick_up = min(self.capacity - self.aboard, self.map[self.position])
        self.aboard += pick_up
        self.map[self.position] -= pick_up
        if self.can_graze(): self.graze()
        else: self.eat()
        while True:
            if self.position == len(self.map) - 1:
                self.map[self.position] += self.aboard
                self.aboard = 0
                break
            if self.can_graze():
                self.graze()
                continue
            if not self.can_sow(): break
            self.sow()
    
    def go_back(self):
        while self.position > self.current_cache:
            self.graze(-1)
    
    def solve(self):
        c.carry_forward()
        c.go_back()
        c.N -= 2
        c.carry_forward()
        c.go_back()
        c.N -=2
        c.carry_forward()
        
        
    def __repr__(self):
        return f"<CamelJourney aboard={self.aboard}, postion={self.position}, map holds {sum(self.map)} bananas>"
        
c = CamelJourney()
c.solve()