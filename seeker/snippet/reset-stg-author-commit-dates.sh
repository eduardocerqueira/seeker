#date: 2023-08-07T16:50:47Z
#url: https://api.github.com/gists/f4cf4c7493f568e09b542a9e34720836
#owner: https://api.github.com/users/jship

for p in $(stg series -P); do stg edit --authdate "$(date '+%Y-%m-%d %H:%M:%S %z')" --committer-date-is-author-date "$p"; done