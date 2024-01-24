#date: 2024-01-24T17:07:59Z
#url: https://api.github.com/gists/a10cabee89a3cf9184511e8ea01b0ded
#owner: https://api.github.com/users/i0x0ff

import struct
import hashlib
import sys
import pprint
import typing as t
from dataclasses import dataclass

import hashlib

from bitcoin import SelectParams
from bitcoin.core import (
    CTransaction,
    CMutableTransaction,
    CMutableTxIn,
    CTxIn,
    CTxOut,
    CScript,
    COutPoint,
    CTxWitness,
    CTxInWitness,
    CScriptWitness,
    COIN,
    lx,
)
from bitcoin.core import script
from bitcoin.wallet import CBech32BitcoinAddress, P2WPKHBitcoinAddress, CBitcoinSecret
from buidl.hd import HDPrivateKey, PrivateKey
from buidl.ecc import S256Point

SelectParams('regtest')

# OP_CTV is ran under OP_NOP4 at the moment
OP_CHECKTEMPLATEVERIFY = script.OP_NOP4

# Helper methods
def sha256(input):
    return hashlib.sha256(input).digest()


def get_txid(tx):
    return tx.GetTxid()[::-1]


def create_template_hash(tx: CTransaction, nIn: int) -> bytes:
    """Most important function, this function takes a transaction and creates an hash which is then evaluated
    by CheckTemplateVerify
    """
    r = b""
    r += struct.pack("<i", tx.nVersion)
    r += struct.pack("<I", tx.nLockTime)
    vin = tx.vin or []
    vout = tx.vout or []

    # vin
    if any(inp.scriptSig for inp in vin):
        r += sha256(b"".join(ser_string(inp.scriptSig) for inp in vin))
    r += struct.pack("<I", len(tx.vin))
    r += sha256(b"".join(struct.pack("<I", inp.nSequence) for inp in vin))
    
    # vout
    r += struct.pack("<I", len(tx.vout))

    r += sha256(b"".join(out.serialize() for out in vout))
    r += struct.pack("<I", nIn)
    return hashlib.sha256(r).digest()


# Define our template and transactions
def hello_world_template(amount: int = None):
    """We call this a transaction template, because it defines vin and vout conditions that must be met in order to
    pass the CTV validation.

    This template in particular only allows donating all the coins to the miners and sending an OP_RETURN
    b"hello world" and can't be spent in any other way.
    """
    tx = CMutableTransaction()
    tx.nVersion = 2
    tx.vout = [CTxOut(amount - amount, CScript([script.OP_RETURN, b"hello world"]))]
    # dummy input, since the coins we're spending here are encumbered solely by CTV and doesn't require any kind of
    # scriptSig. Subsequently, if you look at the vin section of the `create_template_hash` function, it won't affect
    # the hash of this transaction because the `txid` and `index` are not used to calculate the hash.
    tx.vin = [CMutableTxIn()]  # CMutableTxIn has nSequence set to `0xffffffff` by default 
    return tx


def hello_world_tx(amount=None, vin_txid=None, vin_index=None):
    """Take the CTV template and create a finalized transaction by adding proper
    vin information to it.
    """
    tx = hello_world_template(amount)
    # we populate with a proper vin information
    tx.vin = [CTxIn(COutPoint(lx(vin_txid), vin_index), nSequnce=0xffffffff)]
    return tx


def secure_coins_tx(amount: int = None, vin_txid: str = None, vin_index: int = None):
    """Create a transaction that spends the coins from our funding address and send
    them to our OP_CTV address.
    """
    fee = 1000
    template = hello_world_template(amount=amount - fee)
    hello_world_ctv_hash = create_template_hash(template, 0)

    tx = CMutableTransaction()
    
    # set the vin details of the funding addresses utxo, which we want to spend 
    tx.vin = [CTxIn(COutPoint(lx(vin_txid), vin_index))]

    # set the vout with an amount and the destination - this is the part where our coins get locked by OP_CTV
    # opcode
    tx.vout = [CTxOut(amount - fee, CScript([hello_world_ctv_hash, OP_CHECKTEMPLATEVERIFY]))]

    # sign the transaction owned by our funding wallet, so we can spend
    redeem_script = funding_address.to_redeemScript()
    sighash = script.SignatureHash(
        script=redeem_script,
        txTo=tx,
        inIdx=0,
        hashtype=script.SIGHASH_ALL,
        amount=amount,
        sigversion=script.SIGVERSION_WITNESS_V0,
    )
    signature = funding_prvkey.sign(sighash) + bytes([script.SIGHASH_ALL])

    # set witness data
    tx.wit = CTxWitness([CTxInWitness(CScriptWitness([signature, funding_pubkey]))])
    return tx


# Move our funds from funding address into our CTV secured address
ctv_tx = secure_coins_tx(
    amount=int(0.069 * COIN),
    vin_txid="64983f7437eb80e48da7c4178387265d421e1948eee287fb899035f8bba05b4c",
    vin_index=1,
)
print("Serialized tx:", ctv_tx.serialize().hex())


# Spend the CTV transaction
spend_ctv_tx = hello_world_tx(
    amount=int(0.069 * COIN) - 1000,
    vin_txid=get_txid(tx).hex(),
    vin_index=0,
)
print("Serialized tx:", spend_ctv_tx.serialize().hex())