#date: 2021-12-28T16:50:39Z
#url: https://api.github.com/gists/fe9c597fc8b3b89092cac1c35f7d7078
#owner: https://api.github.com/users/ph04

mkdir -p $1

mv $1.sv $1

cd $1

echo "read -sv $1.sv
hierarchy -top $1
proc; opt; techmap; opt
write_json $1.js" > $1.ys

yosys $1.ys $1

netlistsvg $1.js -o $1.svg

ristretto $1.svg