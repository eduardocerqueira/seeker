#date: 2025-08-26T17:07:51Z
#url: https://api.github.com/gists/4095f757375f5b5a4b7721e3afb8a324
#owner: https://api.github.com/users/nazariyv

out=~/Desktop/BerkeleyMonoPatched
rm -rf "$out"; mkdir -p "$out"

for f in ~/Library/Fonts/BerkeleyMono-*.otf; do
  fontforge -script ~/.nerd-fonts/font-patcher \
    --complete --name filename --outputdir "$out" "$f"
done
