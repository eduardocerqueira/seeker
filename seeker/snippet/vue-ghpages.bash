#date: 2023-09-27T16:46:28Z
#url: https://api.github.com/gists/f80cdfe58add7220a867a1f9457ab6ad
#owner: https://api.github.com/users/TSaytson

pnpm build
git add -f dist && git commit -m "dist subtree push"
git subtree push --prefix dist origin gh-pages