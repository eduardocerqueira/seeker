#date: 2025-09-16T16:53:09Z
#url: https://api.github.com/gists/1db69a572dbc4ac9b7183c130e721327
#owner: https://api.github.com/users/nt7nguye

import os
from contextlib import contextmanager
import pdb
from typing import Dict, List
import tigerbeetle

# Test cluster id defaults to 0
TB_CLUSTER_ID = 0
TB_ADDRESS = os.getenv("TB_ADDRESS", "3001")

# DEFAULT ONLY 1 ledger
DEFAULT_LEDGER = 1

# DEFAULT treasury account
TREASURY_ACCOUNT_ID = 1
TREASURY_ACCOUNT_DEBIT_ID = 2
INITIAL_TREASURY_AMOUNT = 1_000_000_000_000

BANK_ACCOUNT_CODE = 1001
CUSTOMER_ACCOUNT_CODE = 1002

# TRANSFER CODES
INITIAL_DEPOSIT_CODE = 2001
TRANSFER_CODE = 2002


@contextmanager
def get_tb_client():
    """Context manager for TigerBeetle client to ensure proper cleanup"""
    client = tigerbeetle.ClientSync(
        cluster_id=TB_CLUSTER_ID, replica_addresses=TB_ADDRESS
    )
    try:
        yield client
    finally:
        client.close()


def create_tb_account(id: int, initial_deposit: int) -> None:
    """Create a single TigerBeetle account"""
    account = tigerbeetle.Account(
        id=id,
        ledger=DEFAULT_LEDGER,
        flags=tigerbeetle.AccountFlags.DEBITS_MUST_NOT_EXCEED_CREDITS,
        code=CUSTOMER_ACCOUNT_CODE,
    )

    with get_tb_client() as client:
        errors = client.create_accounts([account])
        if errors:
            raise ValueError(f"Failed to create account: {errors}")

    # Deposit initial amount into account
    create_tb_transfer(
        id=1_000_000 + id,
        debit_account_id=TREASURY_ACCOUNT_ID,
        credit_account_id=id,
        amount=initial_deposit,
        code=INITIAL_DEPOSIT_CODE,
    )


def create_tb_transfer(
    id: int,
    debit_account_id: int,
    credit_account_id: int,
    amount: int,
    code: int = TRANSFER_CODE,
) -> tigerbeetle.Transfer:
    """Create a single TigerBeetle transfer"""
    transfer = tigerbeetle.Transfer(
        id=id,
        debit_account_id=debit_account_id,
        credit_account_id=credit_account_id,
        amount=amount,
        ledger=DEFAULT_LEDGER,
        code=code,
    )
    with get_tb_client() as client:
        errors = client.create_transfers([transfer])
        if errors:
            raise ValueError(f"Failed to create transfer: {errors}")
    return get_tb_transfer(id)


def get_account_balance(account_id: int) -> int:
    """Get account balance by calculating credits - debits"""
    with get_tb_client() as client:
        accounts = client.lookup_accounts([account_id])
        if not accounts or len(accounts) == 0:
            raise ValueError(f"Account {account_id} not found")
        account = accounts[0]

    return account.credits_posted - account.debits_posted


def get_account_balances(account_ids: List[int]) -> Dict[int, int]:
    """Get account balances for a list of accounts"""
    with get_tb_client() as client:
        accounts = client.lookup_accounts(account_ids)
        return {
            account.id: account.credits_posted - account.debits_posted
            for account in accounts
        }


def get_tb_transfer(transfer_id: int) -> tigerbeetle.Transfer:
    """Get a transfer by ID"""
    with get_tb_client() as client:
        transfers = client.lookup_transfers([transfer_id])
        if not transfers or len(transfers) == 0:
            raise ValueError(f"Transfer {transfer_id} not found")
    return transfers[0]


def get_account_transfer_history(account_id: int) -> List[tigerbeetle.Transfer]:
    """Get transfer history for an account"""
    with get_tb_client() as client:
        # Do both debits and credits
        transfers = client.get_account_transfers(
            tigerbeetle.AccountFilter(
                account_id=account_id,
                limit=100,
                code=TRANSFER_CODE,
                # I got stuck on this for so long :sob:
                # But this is a very very nice API
                flags=tigerbeetle.AccountFilterFlags.DEBITS
                | tigerbeetle.AccountFilterFlags.CREDITS,
            )
        )
        return transfers
