#date: 2023-04-11T17:04:05Z
#url: https://api.github.com/gists/b6640e0ce181748a87e4b53e619596a5
#owner: https://api.github.com/users/Umut-Can-Physics

# Plotting
import matplotlib.pyplot as plt
# Function to set irrational values for alpha
def gcd(a, b): 
    if b == 0: return a
    return gcd(b, a % b)
# Maximum of matrix size (q)
q_max = 50
def plot_butterfly(q_max):
    # Iterations over alpha values
    for p in range(1, q_max+1):
        for q in range(1, q_max+1):
            # Set alpha rational values less than 1
            if q>p:
                if gcd(p, q) == 1:
                    # Define alpha
                    alpha = p/q
                    y = np.zeros(q)
                    y[:] = alpha
                    
                    # Eigenvalues of Harper-Hofstadter matrix for each k values
                    x1 = np.linalg.eigvalsh(H(p,q,kx=0, ky=0))
                    x2 = np.linalg.eigvalsh(H(p,q,kx=np.pi/q, ky=np.pi/q))
                    
                    # Tip (0<k_i<pi/q) of eigenvalues (x1, x2) merge with lines
                    for i in range(len(x1)):
                        plt.plot([x1[i],x2[i]], y[:2], '-', c="black", markersize=0.1)
                    
                    # Plot each energies 
                    plt.plot(x1, y, 'o', c="black", markersize=0.1)
                    plt.plot(x2, y, 'o', c="black", markersize=0.1)

    plt.xlabel(r'$\epsilon$', fontsize=15)
    plt.ylabel(r'$\alpha$', fontsize=15)
    plt.title(r'Hofstadter Butterfly for $q=1-$'+str(q))       
    return 