#date: 2022-12-08T17:05:00Z
#url: https://api.github.com/gists/f5318810e2daae040bbf3958997c07f9
#owner: https://api.github.com/users/mzpqnxow

```
# This won't build a *complete* statically linked smbclient exe, but it will do better ... :/
$ ./configure --without-winbind --without-ldap --without-ads --disable-cups --without-quotas --disable-avahi --without-syslog --without-pam --disable-pthreadpool --without-acl-support --without-automount --without-pie --nopyc --nopyo --hostcc=musl-gcc --disable-python --without-ad-dc --disable-fault-handling --without-libunwind --disable-iprint --without-gettext --disable-python --without-json --with-iconv --without-libarchive --hostcc=musl-gcc --with-static-modules=ALL
```