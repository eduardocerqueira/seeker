#date: 2023-06-02T16:58:47Z
#url: https://api.github.com/gists/a7a69b00f49877c5eb8463ae80939383
#owner: https://api.github.com/users/kondrasso

def euler_method_system(f, x0, y0, z0, h, n):
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]

    for i in range(n):
        x = x_values[-1]
        y = y_values[-1]
        z = z_values[-1]

        dy_dx, dz_dx = f(x, y, z)

        x_next = x + h
        y_next = y + h * dy_dx
        z_next = z + h * dz_dx

        x_values.append(x_next)
        y_values.append(y_next)
        z_values.append(z_next)

    return x_values, y_values, z_values


def f(x, y, z):
    dy_dx = -2 * x * y
    dz_dx = 3 * x - z
    return dy_dx, dz_dx


def main():
    x0 = 0.0
    y0 = 1.0
    z0 = 0.0
    h = 0.1
    n = 10

    x_values, y_values, z_values = euler_method_system(f, x0, y0, z0, h, n)

    # Print the results
    print(f"{'x':<20s}\t\t{'y':<20s}\t\t{'z':<20s}")
    for x, y, z in zip(x_values, y_values, z_values):
        print(f"{x:<20f}\t\t{y:<20f}\t\t{z:<20f}")


if __name__ == "__main__":
    main()
