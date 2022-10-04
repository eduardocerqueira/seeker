#date: 2022-10-04T17:29:37Z
#url: https://api.github.com/gists/abfd1db97ba0149af24c667de7bc6f9b
#owner: https://api.github.com/users/iversond

filename="terraform-provider-oci_v3.85.0_x4"

git filter-branch --prune-empty -d /tmp/scratch \
  --index-filter "git rm --cached -f --ignore-unmatch $filename" \
  --tag-name-filter cat -- --all