#date: 2023-06-26T16:52:16Z
#url: https://api.github.com/gists/f595bc3595ed0f1f825ad8fefb74a369
#owner: https://api.github.com/users/er0p

#!/bin/sh

DEV=$1
SFILE=$2
DFILE=$3
BAUDRATE=$4

usage() {
    scr_name=$(basename $0)
    echo "${scr_name} dev src_f dst_f [ baud_rate ] "
    echo ""
    echo "  char dev, e.g. /dev/ttyUSB0"
    echo "  s_file - file path on local host that be copied to the target(remote)"
    echo "  d_file - file path on target(remote) host (will be created)"
    echo "  most common baudrates: 1200 2400 4800 19200 38400 57600 115200"
    echo ""
    echo "  example:"
    echo "    ./${scr_name} /dev/ttyUSB0 /tmp/src_file.bin /tmp/dst_file.bin 115200"
    echo ""
}

function bin2hex(){
    hexdump -v -e '1/1 "%02x"' "$1"
}

function print_hex2bin_here_doc() {
    echo "======================="
    echo "====== hex2bin ========"
    echo "======================="
cat << END
cat << EOF > ./hex2bin.sh
#!/bin/sh
tmp_file=/tmp/\$1
sed 's/\([0-9A-F]\{2\}\)/\\\\\\x\1/gI' "\$1" > \${tmp_file}
cat \${tmp_file} | awk -v FS="" '{ for (i=1;i<=NF;i+=6) { res=substr(\$0,i,6); system("printf " res ); } }'
EOF
chmod +x ./hex2bin.sh
END
    echo "======================="
}

function print_bin2hex_here_doc() {
    echo "======================="
    echo "====== bin2hex ========"
    echo "======================="
cat << END
cat << EOF > ./bin2hex.sh
#!/bin/sh
hexdump -v -e '1/1 "%02x"' "\$1"
EOF
chmod +x ./bin2hex.sh
END
    echo "======================="
}


if [ -z ${DEV} ] ; then
    echo "device is not specified"
    usage
    exit ${LINENO}
fi

if [ ! -c ${DEV} ] ; then
    echo "device ${DEV} is not char device"
    exit ${LINENO}
fi

if [ -z ${SFILE} ] ; then
    echo "error: src filepath has to be specified"
    usage
    exit ${LINENO}
fi

if [ -z ${DFILE} ] ; then
    echo "error: dest filepath has to be specified"
    usage
    exit ${LINENO}
fi

if [ ! -f ${SFILE} ] ; then
    echo "error: file \"${SFILE}\" isn't a file"
    usage
    exit ${LINENO}
fi

if [ -z ${BAUDRATE} ] ; then
    BAUDRATE=115200
fi

BNAME=$(basename ${DEV})
SNAME=$(screen -ls | grep ${BNAME})
if [  -z "${SNAME}" ] ; then
    cat << EOF

A new screen session ${SNAME} for dev ${DEV} will be created"

   screen -S ${BNAME} ${DEV} ${BAUDRATE}

then detach from there ctrl-a + d"

EOF
    read -s -n 1 -p "Press any key to continue . . ."
    screen -S ${BNAME} ${DEV} ${BAUDRATE}
else
    SNAME=$(echo ${SNAME} | awk '{print $1}' | cut -d'.' -f 2)
    echo "attach to existed session ${SNAME}"
    echo screen -rd ${SNAME}
fi


SNAME=$(basename ${DEV})
RES=$(screen -ls | grep ${SNAME})


SFILE_HEX=${SFILE}".hex"

bin2hex ${SFILE} > ${SFILE_HEX}
echo ""
echo "md5sum ${SFILE}"
md5sum ${SFILE}
echo "md5sum ${SFILE_HEX}"
md5sum ${SFILE_HEX}
echo ""
echo sudo screen -S ${SNAME} -X readreg p ${SFILE_HEX}
echo sudo screen -S ${SNAME} -X stuff "\"cat > ${DFILE} \$(echo -ne '\r')\""
echo sudo screen -S ${SNAME} -X paste p
echo sudo screen -S ${SNAME} -X stuff "\"\$(echo -ne '\r')\""
echo sudo screen -S ${SNAME} -X stuff "\"\$(echo -ne '\x04')\""
echo "cat ${DFILE} | tr -d '\n' > ${DFILE}.tmp"
echo "mv ${DFILE}.tmp ${DFILE}"
echo "md5sum ${DFILE}"


cat << EOF

####################
##### Example: #####
####################

Example:

$ base64 x > y

$ screen -S ttyUSB0 -X readreg p y
$ screen -S ttyUSB0 -X stuff "cat > x \$(echo -ne '\r')"
$ screen -S ttyUSB0 -X paste p
$ screen -S ttyUSB0 -X stuff "\$(echo -ne '\\04 \\r')"
$ screen -S ttyUSB0 -X stuff "base64 -d y > z \$(echo -ne '\\r')"

! ! ! ! ! ! ! ! ! ! !
! ! ! ATTENTION ! ! !
! ! ! ! ! ! ! ! ! ! !
EVERY FILE, THAT IS GOING TO BE TRANSFERED OVER THIS APPROACH MUST BE CONVERTED INTO TEXT FROM ( PURE BINARIES CANNOT BE TRANSFERED USING READREG / PASTE FEATURE OF SCREEN)

SO, IF YOU DO NOT HAVE BASE64 UTILITY ON REMOTE'S SIDE: ( E.G. BUSYBOX ), YOU COULD USE THE FOLLOWING SCRIPT ON REMOTE'S SIDE:
EOF
print_hex2bin_here_doc
echo "AND USE THIS SCRIPT ON HOST'S SIDE:"
print_bin2hex_here_doc
echo "####################"