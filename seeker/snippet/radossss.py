#date: 2023-03-06T17:05:12Z
#url: https://api.github.com/gists/07a84ad5acf9bb0c8e956f7a6204e659
#owner: https://api.github.com/users/cbodley

cbodley@localhost ~/ceph/build $ LD_PRELOAD=/usr/lib64/libasan.so.8 python3.8 -m radossss
/usr/bin/python3.8: No module named radossss

=================================================================
==166101==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 108160 byte(s) in 83 object(s) allocated from:
    #0 0x7f26cb8ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f26cb503419 in PyObject_Malloc (/lib64/libpython3.8.so.1.0+0x103419)

Direct leak of 9802 byte(s) in 13 object(s) allocated from:
    #0 0x7f26cb8ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f26cb504c3e in PyUnicode_New (/lib64/libpython3.8.so.1.0+0x104c3e)

Direct leak of 1242 byte(s) in 2 object(s) allocated from:
    #0 0x7f26cb8ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f26cb510d41 in _PyBytes_FromSize (/lib64/libpython3.8.so.1.0+0x110d41)

Direct leak of 1096 byte(s) in 2 object(s) allocated from:
    #0 0x7f26cb8ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f26cb503305 in _PyObject_GC_Alloc (/lib64/libpython3.8.so.1.0+0x103305)

Direct leak of 536 byte(s) in 1 object(s) allocated from:
    #0 0x7f26cb8ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f26cb50d00c in _PyObject_Realloc.part.0 (/lib64/libpython3.8.so.1.0+0x10d00c)

Indirect leak of 15912 byte(s) in 17 object(s) allocated from:
    #0 0x7f26cb8ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f26cb503305 in _PyObject_GC_Alloc (/lib64/libpython3.8.so.1.0+0x103305)

Indirect leak of 7439 byte(s) in 7 object(s) allocated from:
    #0 0x7f26cb8ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f26cb504c3e in PyUnicode_New (/lib64/libpython3.8.so.1.0+0x104c3e)

Indirect leak of 2571 byte(s) in 2 object(s) allocated from:
    #0 0x7f26cb8ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f26cb503419 in PyObject_Malloc (/lib64/libpython3.8.so.1.0+0x103419)

SUMMARY: AddressSanitizer: 146758 byte(s) leaked in 127 allocation(s).

cbodley@localhost ~/ceph/build $ LD_PRELOAD=/usr/lib64/libasan.so.8 python3.9 -m radossss
/usr/bin/python3.9: No module named radossss

=================================================================
==166104==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 124084 byte(s) in 104 object(s) allocated from:
    #0 0x7f43546ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f435430643e in PyObject_Malloc (/lib64/libpython3.9.so.1.0+0x10643e)

Direct leak of 1 byte(s) in 1 object(s) allocated from:
    #0 0x7f43546ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f435430e04a in PyMem_Malloc (/lib64/libpython3.9.so.1.0+0x10e04a)

Indirect leak of 25862 byte(s) in 26 object(s) allocated from:
    #0 0x7f43546ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f435430643e in PyObject_Malloc (/lib64/libpython3.9.so.1.0+0x10643e)

Indirect leak of 536 byte(s) in 1 object(s) allocated from:
    #0 0x7f43546ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f43543875a5 in _PyObject_Malloc.part.0 (/lib64/libpython3.9.so.1.0+0x1875a5)

SUMMARY: AddressSanitizer: 150483 byte(s) leaked in 132 allocation(s).

cbodley@localhost ~/ceph/build $ LD_PRELOAD=/usr/lib64/libasan.so.8 python3.10 -m radossss
/usr/bin/python3.10: No module named radossss

=================================================================
==166106==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 80907 byte(s) in 78 object(s) allocated from:
    #0 0x7f791d4ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f791d101bf1 in PyObject_Malloc (/lib64/libpython3.10.so.1.0+0x101bf1)

Indirect leak of 4719 byte(s) in 5 object(s) allocated from:
    #0 0x7f791d4ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f791d101bf1 in PyObject_Malloc (/lib64/libpython3.10.so.1.0+0x101bf1)

SUMMARY: AddressSanitizer: 85626 byte(s) leaked in 83 allocation(s).

cbodley@localhost ~/ceph/build $ LD_PRELOAD=/usr/lib64/libasan.so.8 python3.11 -m radossss
/usr/bin/python3.11: No module named radossss

=================================================================
==166116==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 13628 byte(s) in 6 object(s) allocated from:
    #0 0x7f96946ba6af in __interceptor_malloc (/usr/lib64/libasan.so.8+0xba6af)
    #1 0x7f96941a6be2 in PyObject_Malloc (/lib64/libpython3.11.so.1.0+0x1a6be2)

SUMMARY: AddressSanitizer: 13628 byte(s) leaked in 6 allocation(s).

cbodley@localhost ~/ceph/build $ LD_PRELOAD=/usr/lib64/libasan.so.8 python3.12 -m radossss
/usr/bin/python3.12: No module named radossss