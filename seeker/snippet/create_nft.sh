#date: 2021-10-28T17:08:08Z
#url: https://api.github.com/gists/0373fd180a753a492a44d51d3699eb5b
#owner: https://api.github.com/users/cryptorings

# NOTES:
# erd1mzzclada2ly8scj0zrcldvfftfmc9uc42z27c9zgax59k86qsr2qpx0ama is our wallet address; REPLACE this with your wallet address 
# ESDTNFTCreate is the command to create an NFT
# 0x52494e47504153532d656537376234 is the identifier for our NFT collection in hex (RINGPASS-ee77b4)
# 1 is quantity to create (must always be one for NFT)
# 0x52696e672050617373 is the name of the NFT hex encoded (here "Ring Pass")
# 0 is NFT royalties to take
# 0x00ad12b8600c09a844551018255763831173488f33804a461f6009418c34cf07 is the hash of our image
# 0 is NFT attributes (for us, also none)
# 0x68747470733a2f2f676... is the URI for our image on Pinata, encoded as hex


erdpy contract call erd1mzzclada2ly8scj0zrcldvfftfmc9uc42z27c9zgax59k86qsr2qpx0ama --function ESDTNFTCreate --arguments 0x52494e47504153532d656537376234 1 0x52696e672050617373 0 0x00ad12b8600c09a844551018255763831173488f33804a461f6009418c34cf07 0 0x68747470733a2f2f676174657761792e70696e6174612e636c6f75642f697066732f516d6231645563726e724d346953566535357339744b5372344177706931624c693759484b34587848586d623344  --recall-nonce --gas-limit 70000000 --pem WALLET_NAME.pem --send
