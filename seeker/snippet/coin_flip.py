#date: 2024-03-19T16:46:40Z
#url: https://api.github.com/gists/324a6eea5b92af2a38afba1f28519e66
#owner: https://api.github.com/users/maweigert

""" 
Flip a fair coin 100 times—it gives a sequence of heads (H) and tails (T). 
For each HH in the sequence of flips, Alice gets a point; for each HT, Bob does,
so e.g. for the sequence THHHT Alice gets 2 points and Bob gets 1 point. 
Who is most likely to win?


https://x.com/littmath/status/1769044719034647001?s=20


for 100 flips, the probability of Alice winning is 

145031987309855208595272106851/316912650057057350374175801344  ~ 0.4576402

"""

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

def compute_winning_probabilities(n:int, normalize=True):
    """markov chain approach with states H_s and T_s (s = -n ...n), where
    
    state H_D -> a sequence ending in H with score difference D = score(alice) - score(bob)
    state T_D -> a sequence ending in T with score difference D = score(alice) - score(bob)
    
    the transition matrix P_{ij} is then 
    
        H_D -> H_{D+1} with p=0.5  (adding H to H gives alice a point)
        H_D -> T_{D-1} with p=0.5  (adding T to H gives bob a point)
        T_D -> T_D     with p=0.5  (adding T to T gives no points)
        T_D -> H_D     with p=0.5  (adding H to T gives no points)
    
    Winning prob of alice can be inferred from summing all elements with s>0    
    after P^(n-1)@p0, where p0 is the initial state vector with p(H_0)=p(T_0)=0.5
    
    
    """
    # maps i to the index of the score i
    h_n = lambda i: 2*(n+i)
    t_n = lambda i: 2*(n+i)+1
    ns = np.arange(-n, n+1)
    scores = np.repeat(ns, 2)
    # the transition matrix
    P = csr_matrix((4*n+2, 4*n+2), dtype=np.float32)
    for i in ns:
        if i+1<=n:  P[h_n(i+1), h_n(i)] = 0.5
        if i-1>=-n: P[t_n(i-1), h_n(i)] = 0.5
        P[t_n(i), t_n(i)] = 0.5
        P[h_n(i), t_n(i)] = 0.5
    p = np.zeros(4*n+2)
    p[[h_n(0),t_n(0)]] = .5
    for _ in tqdm(range(n-1)):
        p = P@p
    # alice wins if score is positive
    p_alice = p[scores>0].sum()
    p_bob = p[scores<0].sum()
    return p_alice, p_bob


n_flips=100

p_alice, p_bob = compute_winning_probabilities(n_flips)

print(f'\n--- number of flips: {n_flips} ---\n')
print(f"P(Alice wins) = {p_alice:.6f}")
print(f"P(Bob wins)   = {p_bob:.6f}\n")
      
