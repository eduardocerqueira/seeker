#date: 2022-07-04T03:32:08Z
#url: https://api.github.com/gists/1f55eb3ea76c3a8ed16703cc7bd4d322
#owner: https://api.github.com/users/ljmartin

def f(x):
    return (x**3 + x**2 + x + 1) / (x**2 + x + 1)

x = np.linspace(0,10,100)
plt.plot(x, f(x))
plt.plot(x,x)