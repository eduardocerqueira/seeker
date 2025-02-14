#date: 2025-02-14T16:44:13Z
#url: https://api.github.com/gists/6a2ef45b23bbf2a2543b8bb9cc0941a9
#owner: https://api.github.com/users/yashs33244

from solana.rpc.api import Client
from solana.transaction import Transaction
from solana.system_program import TransferParams, transfer
from solana.keypair import Keypair
from solana.publickey import PublicKey
import base58
import base64
from typing import Optional
import logging
from decimal import Decimal

class SolanaTransactionError(Exception):
    """Custom exception for Solana transaction errors"""
    pass

class SecureTransactionSigner:
    def __init__(self, rpc_url: str = "https://api.devnet.solana.com"):
        """
        Initialize the transaction signer with an RPC URL
        
        Args:
            rpc_url: Solana RPC endpoint URL
        """
        self.client = Client(rpc_url)
        self.logger = logging.getLogger(__name__)
        
    def validate_amount(self, amount: float) -> int:
        """
        Validate and convert SOL amount to lamports
        
        Args:
            amount: Amount in SOL
            
        Returns:
            Amount in lamports
            
        Raises:
            ValueError: If amount is invalid
        """
        if not isinstance(amount, (int, float, Decimal)):
            raise ValueError("Amount must be a number")
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")
        if amount > 1000000:  # Sanity check limit
            raise ValueError("Amount exceeds maximum allowed")
        return int(amount * 10**9)

    def create_signed_transaction(
        self,
        sender_private_key: Optional[str] = None,
        recipient_address: str = None,
        amount_sol: float = None,
    ) -> str:
        """
        Create and sign a Solana transfer transaction
        
        Args:
            sender_private_key: Base58 encoded private key
            recipient_address: Recipient's public key in base58
            amount_sol: Amount in SOL
            
        Returns:
            Base64 encoded signed transaction
            
        Raises:
            SolanaTransactionError: If transaction creation fails
            ValueError: If input validation fails
        """
        try:
            # Input validation
            if not all([sender_private_key, recipient_address, amount_sol]):
                raise ValueError("Missing required parameters")

            # Convert amount to lamports
            amount_lamports = self.validate_amount(amount_sol)
            
            # Create sender keypair
            try:
                sender_keypair = "**********"
                    base58.b58decode(sender_private_key)
                )
            except Exception as e:
                raise ValueError(f"Invalid sender private key: {str(e)}")

            # Create recipient public key
            try:
                recipient_pubkey = PublicKey(recipient_address)
            except Exception as e:
                raise ValueError(f"Invalid recipient address: {str(e)}")

            # Get recent blockhash
            try:
                recent_blockhash = self.client.get_latest_blockhash()
                blockhash = recent_blockhash.value.blockhash
            except Exception as e:
                raise SolanaTransactionError(f"Failed to get recent blockhash: {str(e)}")

            # Check sender balance
            try:
                balance = self.client.get_balance(sender_keypair.public_key).value
                if balance < amount_lamports:
                    raise ValueError("Insufficient balance")
            except Exception as e:
                raise SolanaTransactionError(f"Failed to check balance: {str(e)}")

            # Create transaction
            transaction = Transaction()
            
            # Add transfer instruction
            transfer_instruction = transfer(
                TransferParams(
                    from_pubkey=sender_keypair.public_key,
                    to_pubkey=recipient_pubkey,
                    lamports=amount_lamports
                )
            )
            print("transfer_instruction")
            transaction.add(transfer_instruction)
            print("transaction.add()")

            # Set the blockhash and fee payer
            transaction.recent_blockhash = str(blockhash)  # Convert blockhash to string
            print("transaction.recent_blockhash")
            transaction.fee_payer = sender_keypair.public_key
            print("transaction.fee_payer")
            # Sign transaction
            transaction.sign(sender_keypair)
            print("transaction.sign()") 
            # Serialize and encode
            serialized_transaction = transaction.serialize()
            print("serialized_transaction")
            return base64.b64encode(serialized_transaction).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Transaction creation failed: {str(e)}")
            raise

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize signer with devnet
    signer = SecureTransactionSigner("https://api.devnet.solana.com")
    
    try:
        # Get user input securely
        sender_private = input("Enter sender private key: ").strip()
        recipient_public = input("Enter recipient public key: ").strip()
        amount = float(input("Enter amount in SOL: ").strip())
        
        # Create and sign transaction
        signed_txn = signer.create_signed_transaction(
            sender_private_key=sender_private,
            recipient_address=recipient_public,
            amount_sol=amount
        )
        
        # Output base64 encoded transaction
        print("\nSigned transaction (base64):")
        print(signed_txn)
        
    except ValueError as e:
        print(f"Input error: {str(e)}")
    except SolanaTransactionError as e:
        print(f"Transaction error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main():
    main()