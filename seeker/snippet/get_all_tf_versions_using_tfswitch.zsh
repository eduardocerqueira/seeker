#date: 2023-04-26T16:51:41Z
#url: https://api.github.com/gists/f0c73f78939e2ff8600be01c7b00805f
#owner: https://api.github.com/users/r35krag0th

#!/usr/bin/env zsh

declare -a major_minors
major_minors=(
  0.11
  0.12
  0.13
  0.14
  0.15
  1.0
  1.1
  1.2
  1.3
  1.4
)

for short_version in ${major_minors[@]}; do
  LATEST=$(tfswitch -s ${short_version}.0 | grep -E '^Matched version:' | grep -E '(\d.\d{1,}.\d{1,})$' --only-matching | cut -d. -f3)
  echo -e "\033[32m>>>\033[0m Latest for \033[1m${short_version}\033[0m is \033[1m${short_version}.${LATEST}\033[0m"

  for patch_version in {0..${LATEST}}; do
    tfswitch "${short_version}.${patch_version}"
  done
  echo ""
done

echo -e "\033[32m>>>\033[0m Checking code signing on downloaded versions"
find ~/.terraform.versions \
  -name 'terraform_*' \
  -type f \
  -exec codesign -v {} \;
