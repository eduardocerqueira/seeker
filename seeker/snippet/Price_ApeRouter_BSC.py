#date: 2021-09-02T17:05:11Z
#url: https://api.github.com/gists/39772ca81a359a466ffa86455bf07707
#owner: https://api.github.com/users/ismaventuras

from web3 import Web3

#This is the minimum ABI required to get the price from the contract
min_abi = [
        {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "amountIn",
                "type": "uint256"
            },
            {
                "internalType": "address[]",
                "name": "path",
                "type": "address[]"
            }
        ],
        "name": "getAmountsOut",
        "outputs": [
            {
                "internalType": "uint256[]",
                "name": "amounts",
                "type": "uint256[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

## BSC Specifics
rpc = 'https://bsc-dataseed.binance.org'
router_contract = '0xcF0feBd3f17CEf5b47b0cD257aCf6025c5BFf3b7'
wbnb = '0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c' # wbnb
busd = '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56' #busd

## w3  and router contract
w3 = Web3(Web3.HTTPProvider(rpc))
router = w3.eth.contract(
    address=router_contract,
    abi=min_abi)


def price_in_bnb(token):
    #1*10**18 == 1 in wei
    price_bnb = router.functions.getAmountsOut(
        1*10**18, 
        [token,busd]
        ).call()
    return w3.fromWei(price_bnb[1],'ether')


def price_in_busd(token):
    #1*10**18 == 1 in wei
    price_busd = router.functions.getAmountsOut(
        1*10**18, 
        [token,busd] ## we look for liquidity 
        ).call()
    return w3.fromWei(price_busd[1],'ether')


##Examples
#token_to_check = '0xBA26397cdFF25F0D26E815d218Ef3C77609ae7f1' #Lyptus
#print(price_in_bnb(token_to_check))
#print(price_in_busd(token_to_check))