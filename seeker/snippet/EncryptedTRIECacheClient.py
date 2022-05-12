#date: 2022-05-12T17:10:39Z
#url: https://api.github.com/gists/21abd34021c21b75a1d79c111d088dbe
#owner: https://api.github.com/users/anumukhe

from EncryptedTRIECacheImpl import CryptoTrie

outputstrlist = ["Not present in CryptoTrie DataStore","Present in CryptoTrie DataStore"]

def populateCryptoTrie(trienode, keys):
    # Construct trie
    for key in keys:
        trienode.add(key)
    
    print("\ndump content\n")
    trienode.dumpcontent()
    
def searchCryptoTrie(trienode):
    # Search for different keys
    print('\n----------------------------\n')
    print("\nThe word {} ---- {}\n".format("12fine",outputstrlist[trienode.search("12fine")]))
    print("\nThe word {} ---- {}\n".format("Chicken",outputstrlist[trienode.search("Chicken")]))
    print(trienode.getwordsstartwith('Pine'), ' are the words starting with \'Pine\'\n')
    print(trienode.getwordsstartwith('Wh'), ' are the words starting with \'Wh\'\n')
    print('\n----------------------------\n')
    
    
    
root = CryptoTrie()

keys = ["Apple","Orange","Pine","Pineapple","White","Wheat"]

# add the entries to TRIE
populateCryptoTrie(root, keys)

# Now search for desired entry
searchCryptoTrie(root)