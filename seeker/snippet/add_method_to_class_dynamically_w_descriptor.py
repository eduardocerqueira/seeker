#date: 2022-01-10T17:22:24Z
#url: https://api.github.com/gists/4891e88869593fecfafe9d191d1cf0b1
#owner: https://api.github.com/users/robdmc

# Some class definition out of your control
class JaredLewis:
    def loves(self):
        return 'Savage Garden'

# A method you wish the class had
def hates(self):
    return "Randy Travis"
    
# Instantiate the class    
jl = JaredLewis()

# Bind methods in increaingly ugly ways
jl.hates = hates.__get__(jl, JaredLewis)
jl.despises = (lambda self: "Rob's hacks").__get__(jl, JaredLewis)

# The HORROR
print(
    f'Jared Lewis loves {jl.loves()}. '
    f'\nHe hates {jl.hates()}. '
    f'\nBut above all he despises {jl.despises()}'
)