#date: 2023-04-05T16:46:19Z
#url: https://api.github.com/gists/2a9022d3c327ca1606ce2b70f3677f84
#owner: https://api.github.com/users/g00ntar

qemu-system-x86_64 -smp 4 -m 20G -nographic -M pc-q35-jammy --accel tcg \
 -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE.fd \
 -drive if=pflash,format=raw,file=flash1.img \
 -drive file=ubuntu-22.04-minimal-cloudimg-amd64.qcow2,format=qcow2,id=drive0,if=none \
 -device virtio-blk,drive=drive0 \
 -device virtio-net,netdev=network0 \
 -netdev tap,id=network0,ifname=tap0,script=no,downscript=no,vhost=on