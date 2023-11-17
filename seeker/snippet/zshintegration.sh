#date: 2023-11-17T16:51:17Z
#url: https://api.github.com/gists/aa4cb4c3d3d84a74eb911d9c9d894f70
#owner: https://api.github.com/users/killvxk

# insert this snippet into your powerlevel10k config and append 'proxychains' to $POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS
function prompt_proxychains() {
  if [[ "$LD_PRELOAD" == */usr/lib/libproxychains4.so* ]]; then
      p10k segment -f 2 -i '↔' -t "${PROXYCHAINS_ENDPOINT:-proxychains}"
  fi
}