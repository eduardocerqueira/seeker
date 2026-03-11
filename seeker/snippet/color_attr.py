#date: 2026-03-11T17:36:59Z
#url: https://api.github.com/gists/def698626b1af617d7662b51040ccb05
#owner: https://api.github.com/users/pythonhacker

def validate_rgb(val):
    # Validate type
    if type(val) is not int:
        raise ValueError("RGB values must be integers")
    # Validate range            
    if not val in range(0, 256):
        raise ValueError("RGB values must be in range {0...255}")
            
class ColorAttr(Color):
    """ Color class with full attribute validation """

    def __init__(self, r=0, g=0, b=0):
        map(validate_rgb, (r, g, b))
        self.r = r
        self.g = g
        self.b = b
            
    def __setattr__(self, name, value):
        if name in ("r", "g", "b"):
            validate_rgb(value)
        super().__setattr__(name, value)