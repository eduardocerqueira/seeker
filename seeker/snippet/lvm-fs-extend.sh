#date: 2024-07-18T16:56:24Z
#url: https://api.github.com/gists/061047f1bd3c75d960c247195b3bd224
#owner: https://api.github.com/users/marshalljmiller

NEW_DRIVE=/dev/sdb
VG=rl
LV=root

pvcreate "$NEW_DRIVE"
vgextend "$VG" "$NEW_DRIVE"
lvextend --extents +100%FREE --resizefs /dev/mapper/"$VG"-"$LV"