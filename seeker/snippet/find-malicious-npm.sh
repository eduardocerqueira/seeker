#date: 2025-09-08T17:12:07Z
#url: https://api.github.com/gists/52046d419f0873316410eeee9b6b5337
#owner: https://api.github.com/users/guicaulada

#!/bin/bash

# Function to display a spinner
spinner() {
  local pid=$1
  local delay=0.1
  local spinstr='|/-\'
  while kill -0 $pid 2>/dev/null; do
    local temp=${spinstr#?}
    printf " [%c]  " "$spinstr"
    local spinstr=$temp${spinstr%"${temp}"}
    sleep $delay
    printf "\b\b\b\b\b\b"
  done
  printf "    \b\b\b\b"
}

# List of packages and their malicious versions
# Modify this based on what youre lookin for
# ie: "package:1.0.0 1.0.1 2.0.0 3.1.2"
packages=(
  "ansi-styles:6.2.2"
  "debug:4.4.2"
  "chalk:5.6.1"
  "supports-color:10.2.1"
  "strip-ansi:7.1.1"
  "ansi-regex:6.2.1"
  "wrap-ansi:9.0.1"
  "color-convert:3.1.1"
  "color-name:2.0.1"
  "is-arrayish:0.3.3"
  "slice-ansi:7.1.1"
  "color:5.0.1"
  "color-string:2.1.1"
  "simple-swizzle:0.2.3"
  "supports-hyperlinks:4.1.1"
  "has-ansi:6.0.1"
  "chalk-template:1.1.1"
  "backslash:0.2.1"
)

temp_file="/tmp/node_modules_paths.txt"

echo "Searching for all node_modules directories on the system... This may take a while."
find / -type d -name node_modules 2>/dev/null > "$temp_file" & pid=$!
spinner $pid
wait $pid
echo "Search for directories complete."

total=$(wc -l < "$temp_file" | awk '{print $1}')
if [ "$total" -eq 0 ]; then
  echo "No node_modules directories found. Search complete."
  rm -f "$temp_file"
  exit 0
fi

echo "Found $total node_modules directories."
echo "Now checking each one for the specified packages..."
results=""
current=0

while read -r nm_path; do
  ((current++))
  progress=$((current * 100 / total))
  echo -ne "\rProgress: $progress% ($current/$total)"

  for entry in "${packages[@]}"; do
    pkg=$(echo "$entry" | cut -d: -f1)
    vers=$(echo "$entry" | cut -d: -f2)

    pkg_path="$nm_path/$pkg"
    if [ -d "$pkg_path" ] && [ -f "$pkg_path/package.json" ]; then
      version=$(node -e "try { console.log(require('$pkg_path/package.json').version); } catch (e) {}" 2>/dev/null)
      if [ -n "$version" ]; then
        if echo " $vers " | grep -q " $version "; then
          results+="\nFound malicious $pkg version $version at $pkg_path"
        else
          results+="\nFound supposedly safe $pkg version $version at $pkg_path"
        fi
      fi
    fi
  done
done < "$temp_file"

rm -f "$temp_file"

echo ""  # New line after progress bar
if [ -n "$results" ]; then
  echo "Findings:"
  echo -e "$results"
else
  echo "No matching packages found."
fi

echo ""  # New line after findings
echo "Search complete."