#date: 2024-11-22T16:49:38Z
#url: https://api.github.com/gists/cd1df6914b55de50dade8425033d628a
#owner: https://api.github.com/users/goors

#!/bin/bash

while getopts ":v:" opt; do
  case ${opt} in
    
    v )
      vault_name=$OPTARG
      ;;
    
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      exit 1
      ;;
  esac
done

# Verify that all required options are provided
if [[ -z $vault_name ]]; then
  echo "Usage: $0  -v <vault_name>" >&2
  exit 1
fi

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

password=$(grep -oP 'The generated password for the elastic built-in superuser is : "**********"

secret_name= "**********"
secret_value= "**********"

az keyvault secret set --vault-name "$vault_name" --name "$secret_name" --value "$secret_value" "$secret_name" --value "$secret_value"