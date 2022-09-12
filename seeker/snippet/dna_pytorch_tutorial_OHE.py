#date: 2022-09-12T17:08:33Z
#url: https://api.github.com/gists/3ebf777e9781065410ed885858a9ed17
#owner: https://api.github.com/users/erinhwilson

def one_hot_encode(seq):
    """
    Given a DNA sequence, return its one-hot encoding
    """
    # Make sure seq has only allowed bases
    allowed = set("ACTGN")
    if not set(seq).issubset(allowed):
        invalid = set(seq) - allowed
        raise ValueError(f"Sequence contains chars not in allowed DNA alphabet (ACGTN): {invalid}")
        
    # Dictionary returning one-hot encoding for each nucleotide 
    nuc_d = {'A':[1.0,0.0,0.0,0.0],
             'C':[0.0,1.0,0.0,0.0],
             'G':[0.0,0.0,1.0,0.0],
             'T':[0.0,0.0,0.0,1.0],
             'N':[0.0,0.0,0.0,0.0]}
    
    # Create array from nucleotide sequence
    vec=np.array([nuc_d[x] for x in seq])
        
    return vec
  
# look at DNA seq of 8 As
a8 = one_hot_encode("AAAAAAAA")
print("AAAAAA:\n",a8)

# prints:
# AAAAAA:
# [[1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]
# [1. 0. 0. 0.]]

# look at DNA seq of random nucleotides
s = one_hot_encode("AGGTACCT")
print("AGGTACC:\n",s)
print("shape:",s.shape)

# prints:
# AGGTACC:
# [[1. 0. 0. 0.]
# [0. 0. 1. 0.]
# [0. 0. 1. 0.]
# [0. 0. 0. 1.]
# [1. 0. 0. 0.]
# [0. 1. 0. 0.]
# [0. 1. 0. 0.]
# [0. 0. 0. 1.]]
# shape: (8, 4)