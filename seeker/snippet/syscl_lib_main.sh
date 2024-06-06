#date: 2024-06-06T16:49:18Z
#url: https://api.github.com/gists/9dbce6f62f1a548189a85ed6b6788914
#owner: https://api.github.com/users/M0nteCarl0

#bin/sh

icpx -fPIC -c -fsycl sycl-lib.cc
icpx -fsycl -shared sycl-lib.o -o libsycl-lib.so
g++ -o sycl-test test-main.cc -L. -lsycl-lib
./sycl-test