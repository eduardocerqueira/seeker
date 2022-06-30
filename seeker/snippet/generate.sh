#date: 2022-06-30T17:00:57Z
#url: https://api.github.com/gists/ce0e97848ed1259603c158c09d59437a
#owner: https://api.github.com/users/skeeto

#!/bin/sh
set -e

cat <<EOF
.POSIX:
VERSION  = 0.2
CC       = cc
CFLAGS   = -std=gnu18 -Wall -Wextra -O3
LDFLAGS  = -s
LDLIBS   = -lncursesw -lsqlite3 -lcurl -lexpat -lgumbo -lyajl
FEATURES = \\
 -DNEWSRAFT_FORMAT_SUPPORT_ATOM10 \\
 -DNEWSRAFT_FORMAT_SUPPORT_RSS \\
 -DNEWSRAFT_FORMAT_SUPPORT_RSSCONTENT \\
 -DNEWSRAFT_FORMAT_SUPPORT_DUBLINCORE \\
 -DNEWSRAFT_FORMAT_SUPPORT_MEDIARSS \\
 -DNEWSRAFT_FORMAT_SUPPORT_YANDEX \\
 -DNEWSRAFT_FORMAT_SUPPORT_RBCNEWS \\
 -DNEWSRAFT_FORMAT_SUPPORT_ATOM03 \\
 -DNEWSRAFT_FORMAT_SUPPORT_GEORSS \\
 -DNEWSRAFT_FORMAT_SUPPORT_GEORSS_GML \\
 -DNEWSRAFT_FORMAT_SUPPORT_JSONFEED

all: newsraft

clean:
	rm -f newsraft \$(obj)

.c.o:
	\$(CC) -c -Isrc -DNEWSRAFT_VERSION='"\$(VERSION)"' \$(CFLAGS) \$(FEATURES) -o \$@ \$<

EOF

obj=
for c in $(find src -name '*.c'); do
    o="${c%%.c}.o" 
    obj="$obj $o"
    cc -Isrc -MM -MT "$o" "$c"
done
printf 'obj ='
printf ' \\\n  %s' $obj
printf '\n\n'

cat <<EOF
newsraft: \$(obj)
	\$(CC) \$(LDFLAGS) -o \$@ \$(obj) \$(LDLIBS)
EOF
