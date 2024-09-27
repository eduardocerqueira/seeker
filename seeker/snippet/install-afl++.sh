#date: 2024-09-27T16:51:21Z
#url: https://api.github.com/gists/b9d320093968be7cd3b4cbd56a091e11
#owner: https://api.github.com/users/a1g3

apt update
apt upgrade -y
apt install build-essential python3-pip automake autoconf gcc-13-plugin-dev unzip clang -y 
wget https://github.com/AFLplusplus/AFLplusplus/archive/refs/tags/v4.21c.zip
bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
unzip v4.21c.zip
cd AFLplusplus-4.21c
LLVM_CONFIG=llvm-config-18 make
make install