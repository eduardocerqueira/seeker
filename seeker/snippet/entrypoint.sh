#date: 2025-04-25T16:50:13Z
#url: https://api.github.com/gists/ca4482ced33a14d7e1a5dea46fec6af8
#owner: https://api.github.com/users/IgorRSGraziano

#!/bin/sh
set -e

#In build stage, we need set variable value like the key
#Example: NEXT_PUBLIC_API_URL=NEXT_PUBLIC_API_URL
#In docker, define this in your Dockerfile
#Like ENTRYPOINT ["/usr/app/entrypoint.sh"]

escape_special_chars() {
  input="$1"
  printf '%s\n' "$input" | sed 's/[&/]/\\&/g'
}

# Replace env variable placeholders with real values
printenv | grep NEXT_PUBLIC_ | while read -r line ; do
  key=$(echo $line | cut -d "=" -f1)
  value=$(echo $line | cut -d "=" -f2-)

  escaped_value=$(escape_special_chars "$value")

  find /usr/app/.next/ -type f -exec sed -i "s|${key}|$escaped_value|g" {} \;
done

# Execute the container's main process (CMD in Dockerfile)
exec "$@"