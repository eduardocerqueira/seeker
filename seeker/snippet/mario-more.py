#date: 2025-02-17T17:10:19Z
#url: https://api.github.com/gists/47e676b7354c8d27f2882144f6e8babd
#owner: https://api.github.com/users/Tyuris-creator

def main():
    while True:
        try:
            height = int(input("Height: "))
            if height > 8 or height < 1:
                raise ValueError
            else:
                break
        except ValueError:
            pass

    draw_pyramide(height, height)


def draw_pyramide(n, z):
    if n == 0:
        return

    draw_pyramide((n-1), z)
    print(" " * (z - n), end='')
    for i in range(n):
        print("#", end='')
    print(" " * 2, end='')
    for i in range(n):
        print("#", end='')
    print()


main()
