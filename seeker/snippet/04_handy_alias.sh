#date: 2022-07-29T17:12:01Z
#url: https://api.github.com/gists/ac2d6234f9c91957b7f728fbc8dd5f2f
#owner: https://api.github.com/users/skybleu

# To make DX a bit better you can alias your installations:
nvm alias arm stable
# arm -> stable (-> v15.6.0)
nvm alias intel lts/fermium
# intel -> lts/fermium (-> v14.15.4)

# To test aliases:
nvm use arm
# Now using node v15.6.0 (npm v7.4.0)
node -e 'console.log(process.arch)'
-> arm64
nvm use intel
# Now using node v14.15.4 (npm v6.14.10)
node -e 'console.log(process.arch)'
-> x64