#date: 2025-01-13T16:48:53Z
#url: https://api.github.com/gists/656dbddd0a9361d3200bb5de68b617e1
#owner: https://api.github.com/users/beau-SC-dev

from xrpl.wallet import Wallet
from xrpl.clients import JsonRpcClient
from xrpl.models.transactions import TrustSet
from xrpl.models.requests import AccountLines
from xrpl.models.amounts.issued_currency_amount import IssuedCurrencyAmount
from xrpl.transaction import submit_and_wait
from typing import Dict

client = JsonRpcClient("https://xrplcluster.com/")
TRUSTLINE_LIMIT = 100_000_000
PFT_ISSUER = "rnQUEEg8yyjrwk9FhyXpKavHyCRJM9BDMW"

def check_for_existing_pft_trustline(wallet_credentials: Wallet) -> bool:
    """Check if a trustline to the PFT token already exists.
     
    Args:
        wallet_credentials (Wallet): The wallet credentials to check the trustline.
        
    Returns:
        bool: True if the trustline exists, False otherwise.
    """
    try:
        account_lines_request = AccountLines(
            account=wallet_credentials.address,
            peer=PFT_ISSUER
        )

        response = client.request(account_lines_request)
        if not response.is_successful():
            print(f"Failed to get account lines: {response}")
            return False
        
        return any(line.get('currency') == 'PFT' for line in response.result.get('lines', []))
    except Exception as e:
        print(f"Error checking trustline: {e}")
        return False

def establish_and_verify_pft_trustline(wallet_credentials: Wallet) -> Dict[str, str]:
    """Establish a trustline to the PFT token and verify its creations.
     
    Args:
        wallet_credentials (Wallet): The wallet credentials to establish the trustline.
        
    Returns:
        dict: A dictionary containing the result of the operation or error details.
    """
    try:
        trustline_precheck = check_for_existing_pft_trustline(wallet_credentials)
        if trustline_precheck:
            return {"status": "success", "message": "Trustline already exists"}
        
        trust_set_txn = TrustSet(
            account=wallet_credentials.address,
            limit_amount=IssuedCurrencyAmount(
                currency="PFT",
                issuer=PFT_ISSUER,
                value=TRUSTLINE_LIMIT,
            )
        )

        response = submit_and_wait(trust_set_txn, client, wallet_credentials)
        if not response.is_successful():
            return {"status": "error", "message": f"Failed to establish trustline: {response.result}"}
        
        trustline_postcheck = check_for_existing_pft_trustline(wallet_credentials)
        if trustline_postcheck:
            return {"status": "success", "message": "Trustline successfully established"}
        else:
            return {"status": "error", "message": "Trustline creation failed"}
    except Exception as e:
        return {"status": "error", "message": f"Error establishing trustline: {e}"}

if __name__ == "__main__":
    wallet = Wallet.create() # New wallet for example, in production should use an already-funded one otherwise the trustline creation will fail
    result = establish_and_verify_pft_trustline(wallet)
    print(result)