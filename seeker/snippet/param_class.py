#date: 2023-02-01T17:06:23Z
#url: https://api.github.com/gists/cc9b4bf243ac504c90bd02ace74d7a29
#owner: https://api.github.com/users/flo-schu


# also in toopy
class Param(float):
    def __new__(self, value, min=None, max=None, step=None) -> float:
        return float.__new__(self, value)
        
    def __init__(self, value, min=None, max=None, step=None):
            float.__init__(value)
            self.min = float(value / 4 if min is None else min)
            self.max = float(value * 2 if max is None else max)
            self.step = float(1 if step is None else step)

    def __mul__(self, other):
        return Param(float(self) * other, **self.__dict__)

    def __div__(self, other):
        return Param(float(self) / other, **self.__dict__)

    def __truediv__(self, other):
        return Param(float(self) / other, **self.__dict__)

    def __add__(self, other):
        return Param(float(self) + other, **self.__dict__)

    def __sub__(self, other):
        return Param(float(self) - other, **self.__dict__)

    def __repr__(self):
        return f"<Param: {float(self)} [{self.min}-{self.max}] step={self.step}>"
