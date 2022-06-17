#date: 2022-06-17T17:01:00Z
#url: https://api.github.com/gists/ea8e93511ddcc4286a3e11f5d4ae133f
#owner: https://api.github.com/users/dfghjkjhgr

"""An attempt to use as many 3.8 features as possible"""

def cool_function(a, b, /):
    """A simple function that does nothing but use a whole bunch of 3.8 features"""
    cool_dict = {}
    for i in range(3):
        cool_dict[i] = f'cool function!1!! Also, {i + (a * b) =}!'
    return (reversed(cool_dict), cool_dict)

def main():
    """Main function"""
    for value in (hello := cool_function(8, 8)):
        print(value)
    print(hello)

if __name__ == "__main__":
    main()