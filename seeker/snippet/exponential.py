#date: 2021-09-27T17:01:58Z
#url: https://api.github.com/gists/9dffdef32f3ac5689d9659c174f45f80
#owner: https://api.github.com/users/codistwa

# ============================================================
# Display of the function
# ============================================================

x = np.linspace(-3,3,100)

plt.plot(x,np.exp(x))
plt.xlabel('x')
plt.ylabel('$e^x$')

plt.grid()

plt.title('Exponential', fontsize=12)
plt.savefig('exponential', bbox_inches='tight')

plt.show()