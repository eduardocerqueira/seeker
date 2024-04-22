#date: 2024-04-22T17:01:49Z
#url: https://api.github.com/gists/280dc05f603e5de25e63b44342d13f24
#owner: https://api.github.com/users/danishsalim

creating file structure:
/linux/drill1$ mkdir hello
/linux/drill1$ cd hello
/linux/drill1/hello$ mkdir five one
/linux/drill1/hello$ ls
five  one
/linux/drill1/hello$ cd five
/linux/drill1/hello/five$ mkdir six
/linux/drill1/hello/five$ ls
six
/linux/drill1/hello/five$ cd six
/linux/drill1/hello/five/six$ touch c.txt
/linux/drill1/hello/five/six$ ls
c.txt
/linux/drill1/hello/five/six$ mkdir seven
/linux/drill1/hello/five/six$ ls
c.txt  seven
/linux/drill1/hello/five/six$ cd seven
/linux/drill1/hello/five/six/seven$ touch error.log
/linux/drill1/hello/five/six/seven$ ls
error.log
/linux/drill1/hello/five/six/seven$ cd ../../..
/linux/drill1/hello$ ls
five  one
/linux/drill1/hello$ cd one
/linux/drill1/hello/one$ touch a.txt b.txt
/linux/drill1/hello/one$ ls
a.txt  b.txt
/linux/drill1/hello/one$ mkdir two
/linux/drill1/hello/one$ cd two
/linux/drill1/hello/one/two$ ls
/linux/drill1/hello/one/two$ touch d.txt
/linux/drill1/hello/one/two$ ls
d.txt
/linux/drill1/hello/one/two$ mkdir three
/linux/drill1/hello/one/two$ cd three
/linux/drill1/hello/one/two/three$ touch e.txt
/linux/drill1/hello/one/two/three$ ls
e.txt
/linux/drill1/hello/one/two/three$ mkdir four
/linux/drill1/hello/one/two/three$ ls
e.txt  four
/linux/drill1/hello/one/two/three$ cd four
/linux/drill1/hello/one/two/three/four$ touch access.log
/linux/drill1/hello/one/two/three/four$ ls
access.log
danish@danish-Inspiron-5559:~/Mountblue/linux/drill1$ tree /home/danish/Mountblue/linux/drill1
/home/danish/Mountblue/linux/drill1
└── hello
    ├── five
    │   └── six
    │       ├── c.txt
    │       └── seven
    │           └── error.log
    └── one
        ├── a.txt
        ├── b.txt
        └── two
            ├── d.txt
            └── three
                ├── e.txt
                └── four
                    └── access.log

for deleting file with ".log" extension:
find /home/danish/Mountblue/linux/drill1/hello -name '*.log'

danish@danish-Inspiron-5559:~/Mountblue/linux/drill1$ tree /home/danish/Mountblue/linux/drill1
/home/danish/Mountblue/linux/drill1
└── hello
    ├── five
    │   └── six
    │       ├── c.txt
    │       └── seven
    └── one
        ├── a.txt
        ├── b.txt
        └── two
            ├── d.txt
            └── three
                ├── e.txt
                └── four
for adding context to a.txt
cd hello/one/
echo "Unix is a family of multitasking, multiuser computer operating systems that derive from the original AT&T Unix, development starting in the 1970s at the Bell Labs research center by Ken Thompson, Dennis Ritchie, and others
" >> a.txt 

=>to delete durectory five 
    rm -R five
=> Rename the one directory to uno
   mv one uno
=> Move a.txt to the two directory
   cd uno
   mv a.txt /home/danish/Mountblue/linux/drill1/hello/uno/two/a.txt


