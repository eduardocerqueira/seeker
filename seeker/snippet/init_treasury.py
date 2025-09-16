#date: 2025-09-16T16:53:09Z
#url: https://api.github.com/gists/1db69a572dbc4ac9b7183c130e721327
#owner: https://api.github.com/users/nt7nguye

import tigerbeetle

from tb_client import (
    BANK_ACCOUNT_CODE,
    DEFAULT_LEDGER,
    INITIAL_TREASURY_AMOUNT,
    TREASURY_ACCOUNT_DEBIT_ID,
    TREASURY_ACCOUNT_ID,
    get_tb_client,
)

if __name__ == "__main__":
    errors = None
    # I think I have to make 2 accounts to do a genesis transaction in tigerbeetle
    with get_tb_client() as client:
        print("Creating treasury accounts")
        try:
            client.lookup_accounts([TREASURY_ACCOUNT_ID, TREASURY_ACCOUNT_DEBIT_ID])
        except Exception:
            # Only create accounts if they don't exist
            errors = client.create_accounts(
                [
                    tigerbeetle.Account(
                        id=TREASURY_ACCOUNT_ID,
                        credits_posted=0,
                        ledger=DEFAULT_LEDGER,
                        flags=tigerbeetle.AccountFlags.DEBITS_MUST_NOT_EXCEED_CREDITS,
                        code=BANK_ACCOUNT_CODE,
                    ),
                    tigerbeetle.Account(
                        id=TREASURY_ACCOUNT_DEBIT_ID,
                        credits_posted=0,
                        ledger=DEFAULT_LEDGER,
                        flags=tigerbeetle.AccountFlags.CREDITS_MUST_NOT_EXCEED_DEBITS,
                        code=BANK_ACCOUNT_CODE,
                    ),
                ]
            )
            if errors:
                raise ValueError(f"Failed to create treasury account: {errors}")

    # Create a genesis transfer
    with get_tb_client() as client:
        print("Creating genesis transfer")
        errors = client.create_transfers(
            [
                tigerbeetle.Transfer(
                    id=1,
                    debit_account_id=TREASURY_ACCOUNT_DEBIT_ID,
                    credit_account_id=TREASURY_ACCOUNT_ID,
                    amount=INITIAL_TREASURY_AMOUNT,
                    ledger=DEFAULT_LEDGER,
                    code=BANK_ACCOUNT_CODE,
                ),
            ]
        )
        if errors:
            raise ValueError(f"Failed to create genesis transfer: {errors}")

    print("Treasury initialized successfully")
