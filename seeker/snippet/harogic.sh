#date: 2024-08-14T18:45:11Z
#url: https://api.github.com/gists/9d7fa4f2ac887e1f7da1906e478ee322
#owner: https://api.github.com/users/nemanjan00

ln -s /usr/lib64/libicuuc.so.75  ../lib/libicuuc.so.60
ln -s /usr/lib64/libicui18n.so.75 ../lib/libicui18n.so.60
rm -rf ../lib/libQt5*
export LD_LIBRARY_PATH=`pwd`:../lib
export QT_QPA_PLATFORM=xcb
./SAStudio4