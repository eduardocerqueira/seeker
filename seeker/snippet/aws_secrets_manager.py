#date: 2025-02-14T17:02:08Z
#url: https://api.github.com/gists/9a833b583d125e2e75f7bb14c0ecf62d
#owner: https://api.github.com/users/juanecl

import boto3
import json
from botocore.exceptions import BotoCoreError, ClientError

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"S "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********"M "**********"a "**********"n "**********"a "**********"g "**********"e "**********"r "**********"C "**********"l "**********"i "**********"e "**********"n "**********"t "**********": "**********"
    """
    AWS Secrets Manager Client

    Usage example:
    secrets_client = "**********"="us-east-1")

    # Create or update a secret
    secrets_client.create_or_update_secret("my_secret", "MySecurePassword123!")

    # Get a secret
    secret_value = "**********"
    print(f"🔑 Secret value: "**********"

    # List secrets
    secrets = "**********"
    print(f"📋 List of secrets: "**********"

    # Delete a secret (with recovery)
    secrets_client.delete_secret("my_secret")

    # Restore a deleted secret
    secrets_client.restore_secret("my_secret")
    """

    def __init__(self, region="us-east-1"):
        """
        Initializes the AWS Secrets Manager client.
        : "**********": AWS region where the secrets are stored.
        """
        self.client = "**********"=region)

    def _handle_client_error(self, error, action):
        """
        Handles client errors and prints a formatted message.
        :param error: The exception raised.
        :param action: The action being performed when the error occurred.
        """
        if isinstance(error, self.client.exceptions.ResourceNotFoundException):
            print(f"❌ The resource was not found during {action}.")
        else:
            print(f"⚠️ Error during {action}: {str(error)}")

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"c "**********"r "**********"e "**********"a "**********"t "**********"e "**********"_ "**********"o "**********"r "**********"_ "**********"u "**********"p "**********"d "**********"a "**********"t "**********"e "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"v "**********"a "**********"l "**********"u "**********"e "**********") "**********": "**********"
        """
        Creates or updates a secret in AWS Secrets Manager.
        : "**********": Name of the secret.
        : "**********": Value of the secret.
        """
        try:
            # Try to get the existing secret
            self.client.get_secret_value(SecretId= "**********"
            print(f"🔄 The secret '{secret_name}' already exists. It will be updated...")
            
            response = "**********"
                SecretId= "**********"
                SecretString=json.dumps({"password": "**********"
            )
            print(f"✅ The secret was updated: "**********"

        except self.client.exceptions.ResourceNotFoundException:
            print(f"🆕 Creating the new secret '{secret_name}'...")

            response = "**********"
                Name= "**********"
                SecretString=json.dumps({"password": "**********"
            )
            print(f"✅ The secret was created: "**********"

        except (BotoCoreError, ClientError) as e:
            self._handle_client_error(e, "creating or updating the secret")

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"n "**********"a "**********"m "**********"e "**********") "**********": "**********"
        """
        Retrieves the value of a secret.
        : "**********": Name of the secret.
        : "**********": Value of the secret or None if it does not exist.
        """
        try:
            response = "**********"=secret_name)
            secret_value = "**********"
            return secret_value
        except (BotoCoreError, ClientError) as e:
            self._handle_client_error(e, "retrieving the secret")
            return None

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"d "**********"e "**********"l "**********"e "**********"t "**********"e "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"f "**********"o "**********"r "**********"c "**********"e "**********"_ "**********"d "**********"e "**********"l "**********"e "**********"t "**********"e "**********"= "**********"F "**********"a "**********"l "**********"s "**********"e "**********") "**********": "**********"
        """
        Deletes a secret with the option to permanently delete it.
        : "**********": Name of the secret.
        : "**********": If True, deletes the secret without a recovery period.
        """
        try:
            if force_delete:
                response = "**********"=secret_name, ForceDeleteWithoutRecovery=True)
                print(f"💀 The secret was permanently deleted: "**********"
            else:
                response = "**********"=secret_name)
                print(f"🗑️ The secret was moved to deletion with recovery: "**********"

            return response
        except (BotoCoreError, ClientError) as e:
            self._handle_client_error(e, "deleting the secret")

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"l "**********"i "**********"s "**********"t "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
        """
        Lists all secrets stored in AWS Secrets Manager.
        : "**********": List of secret names.
        """
        try:
            secrets = "**********"
            response = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********". "**********"g "**********"e "**********"t "**********"( "**********"" "**********"S "**********"e "**********"c "**********"r "**********"e "**********"t "**********"L "**********"i "**********"s "**********"t "**********"" "**********", "**********"  "**********"[ "**********"] "**********") "**********": "**********"
                secrets.append(secret["Name"])

            return secrets
        except (BotoCoreError, ClientError) as e:
            self._handle_client_error(e, "listing the secrets")
            return []

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"r "**********"e "**********"s "**********"t "**********"o "**********"r "**********"e "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"n "**********"a "**********"m "**********"e "**********") "**********": "**********"
        """
        Restores a secret that has been marked for deletion.
        : "**********": Name of the secret to restore.
        """
        try:
            response = "**********"=secret_name)
            print(f"♻️ The secret was restored: "**********"
        except (BotoCoreError, ClientError) as e:
            self._handle_client_error(e, "restoring the secret")