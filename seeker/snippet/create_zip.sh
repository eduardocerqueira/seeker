#date: 2024-02-14T16:56:02Z
#url: https://api.github.com/gists/a8ce181471020270ab934213eaf646ac
#owner: https://api.github.com/users/Eric-Uncork-it

#!/bin/bash

zipfile="${npm_package_name}v${npm_package_version}.zip"
# Initialize the zip file to ensure it's empty
rm -f "$zipfile"

# List your files and directories
items=(
  "dist"
  "prisma"
  "node_modules"
  ".env"
  "run.js"
  "package.json"
  "package-lock.json"
  "readme.md"
  "tsconfig.json"
)

for item in "${items[@]}"; do
  if [ -e "$item" ]; then
    # Add the item to the zip file if it exists
    zip -r "$zipfile" "$item"
  else
    echo "$item does not exist, skipping..."
  fi
done