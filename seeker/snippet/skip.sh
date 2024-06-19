#date: 2024-06-19T17:00:44Z
#url: https://api.github.com/gists/28d4be593eb4727d2089eff80e86a31e
#owner: https://api.github.com/users/dims

gh api \
  --method POST \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -f 'state=success' \
  -f 'target_url=https://github.com/google/cadvisor/pull/3527/checks' \
  -f 'description=Skipped this stale build!'  \
  -f 'context=test-integration (1.22, ubuntu-20.04, build/config/plain.sh)' \
  /repos/google/cadvisor/statuses/2b6f92b40593f935ca19306af5c3aee3c00ac61d

