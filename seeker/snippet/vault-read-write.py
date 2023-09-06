#date: 2023-09-06T16:56:45Z
#url: https://api.github.com/gists/0fdac4a734e70c58adca593486703642
#owner: https://api.github.com/users/devops-school

import hvac

# Initialize the Vault client
 "**********"d "**********"e "**********"f "**********"  "**********"i "**********"n "**********"i "**********"t "**********"i "**********"a "**********"l "**********"i "**********"z "**********"e "**********"_ "**********"v "**********"a "**********"u "**********"l "**********"t "**********"_ "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********"( "**********"v "**********"a "**********"u "**********"l "**********"t "**********"_ "**********"a "**********"d "**********"d "**********"r "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    client = "**********"=vault_addr, token=token)
    return client

# Function to store a password in Vault
 "**********"d "**********"e "**********"f "**********"  "**********"i "**********"n "**********"s "**********"e "**********"r "**********"t "**********"( "**********"v "**********"a "**********"u "**********"l "**********"t "**********"_ "**********"a "**********"d "**********"d "**********"r "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    try:
        client = "**********"
        # Write the password to Vault
        client.secrets.kv.v2.create_or_update_secret(
            path= "**********"
            secret= "**********"=password)
        )
        return True, "Password stored successfully"
    except Exception as e:
        return False, f"Error storing password: "**********"

# Function to read a password from Vault using a token
 "**********"d "**********"e "**********"f "**********"  "**********"d "**********"i "**********"s "**********"p "**********"l "**********"a "**********"y "**********"( "**********"v "**********"a "**********"u "**********"l "**********"t "**********"_ "**********"a "**********"d "**********"d "**********"r "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    try:
        client = "**********"
        # Read the password from Vault
        result = "**********"='passwords')
        if result is not None and 'data' in result:
            password = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
                return True, password
        return False, "Password not found"
    except Exception as e:
        return False, f"Error reading password: "**********"

# Example usage
if __name__ == "__main__":
    vault_addr = "http://127.0.0.1:8200"
    token = "**********"
    password = "**********"

    # Store the password
    success, message = "**********"
    print(message)

    # Retrieve the password
    success, retrieved_password = "**********"
    if success:
        print(f"Retrieved password: "**********"
    else:
        print(retrieved_password)