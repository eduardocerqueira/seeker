#date: 2021-11-04T17:10:18Z
#url: https://api.github.com/gists/4822c694b8e26cb59d9de19ba7cee8dd
#owner: https://api.github.com/users/jummy123

from brownie import *

JOETROLLER_ADDRESS = 'dc13687554205E5b89Ac783db14bb5bba4A1eDaC'

JAVAX_ADDRESS = 'C22F01ddc8010Ee05574028528614634684EC29e'


def main():
    print('starting account balance: ', accounts[0].balance())
    contract = FlashloanBorrower.deploy(
        JOETROLLER_ADDRESS,
        {'from': accounts[0]}
    )   
    print('Contract deployed address: ', contract)

    contract.deposit({'from': accounts[0], 'value': 2e18})
    print('contract balance ', contract.balance())
    print('account[0] balance ', accounts[0].balance())

    transaction = contract.doFlashloan(
        JAVAX_ADDRESS,
        200000000,
        {'from': accounts[0]})
    print('AVAX flash loan transaction sent: ', transaction.info())
