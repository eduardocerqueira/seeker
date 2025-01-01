#date: 2025-01-01T17:04:19Z
#url: https://api.github.com/gists/6f863d0f3d8f8b60192bd58893e9d450
#owner: https://api.github.com/users/PieroPaialungaAI

plt.figure(figsize=(10,5))
x = np.linspace(-2*np.pi,2*np.pi,1000)
y = np.sin(x) + 0.4*np.cos(2*x) + 2*np.sin(3.2*x)
plt.plot(x,y,color='firebrick',lw=2)
plt.xlabel('Time (t)',fontsize=24)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.4)
plt.ylabel('y',fontsize=24)