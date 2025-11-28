#date: 2025-11-28T16:51:55Z
#url: https://api.github.com/gists/9db185943e0acbd72f4c9bd5375c4b40
#owner: https://api.github.com/users/BabakSamimi

cd emacs
make clean
make -j$(nproc)
sudo make install
cd ..