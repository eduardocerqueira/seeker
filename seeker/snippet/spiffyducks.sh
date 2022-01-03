#date: 2022-01-03T17:14:17Z
#url: https://api.github.com/gists/cd3f27886f9128039196a75cc5884646
#owner: https://api.github.com/users/danfunk

git clone https://github.com/sartography/SpiffyDucks.git
cd SpiffyDucks
python3 -m venv venv
source ./venv/bin/activate
pip3 install spiffworkflow
python ducks.py