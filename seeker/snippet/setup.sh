#date: 2023-01-19T16:47:40Z
#url: https://api.github.com/gists/7f444e6cf42354925e5b3df75d03de24
#owner: https://api.github.com/users/WhistlingZephyr

git clone https://github.com/rust-lang/crates.io-index
rm crates.io-index/config.json
cd crates.io-index
(for i in **/*; do [ -f "$i" ] && echo $i; done) > ../files
cd ..
node generate-tree.js
