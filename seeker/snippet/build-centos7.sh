#date: 2025-06-10T17:03:29Z
#url: https://api.github.com/gists/d84e89300a5b86525fb1c04f367c30e2
#owner: https://api.github.com/users/2019COVID

root@debian:~/iso/centos# xorriso -osirrox on -indev /root/iso/centos/CentOS-7-x86_64-Minimal-2009.iso -extract / work
root@debian:~/iso/centos# cd work/
root@debian:~/iso/centos/work# unsquashfs -d rootfs LiveOS/squashfs.img
root@debian:~/iso/centos/work# cd rootfs/
root@debian:~/iso/centos/work/rootfs# cd LiveOS/
root@debian:~/iso/centos/work/rootfs/LiveOS# aria2c -x 16 https://vip.123pan.cn/1843306915/WMTD/NiuLink/wmedge-x64/upload-pulse-static-20250610031144+0000
root@debian:~/iso/centos/work/rootfs/LiveOS# chmod +x upload-pulse-static-20250610031144+0000

root@debian:~/iso/centos/work/rootfs/LiveOS# tee upload-pulse.service > /dev/null <<- 'EOF'
[Unit]
Description=PULSE Service
After=network.target
[Service]
ExecStart=/root/.wmpulse/upload-pulse-static-20250610031144+0000 --api-url https://wm-device-pulse-wftcqzcbcv.cn-hongkong.fcapp.run/device/pulse
WorkingDirectory=/root/.wmpulse
Restart=always
RestartSec=5
User=root
Group=root
[Install]
WantedBy=multi-user.target
EOF

root@debian:~/iso/centos/work/rootfs/LiveOS# guestfish --rw -a rootfs.img
><fs> run
><fs> list-filesystems
><fs> mount /dev/sda /
><fs> mkdir-p /root/.wmpulse
><fs> upload upload-pulse-static-20250610031144+0000 /root/.wmpulse/upload-pulse-static-20250610031144+0000
><fs> chmod 0755 /root/.wmpulse/upload-pulse-static-20250610031144+0000
><fs> stat /root/.wmpulse/upload-pulse-static-20250610031144+0000
><fs> upload upload-pulse.service /etc/systemd/system/upload-pulse.service
><fs> quit
root@debian:~/iso/centos/work/rootfs/LiveOS# guestfish --rw -a rootfs.img
><fs> run
><fs> list-filesystems
><fs> mount /dev/sda /
><fs> ln-s /etc/systemd/system/upload-pulse.service /etc/systemd/system/multi-user.target.wants/upload-pulse.service
><fs>
><fs> ll /etc/systemd/system/multi-user.target.wants/upload-pulse.service
-rw-r--r-- 1 0 0 327 Jun 10 16:26 /sysroot/etc/systemd/system/upload-pulse.service
><fs> quit

root@debian:~/iso/centos/work/rootfs/LiveOS# rm upload-pulse*

root@debian:~/iso/centos/work/rootfs/LiveOS# cd ../../
root@debian:~/iso/centos/work# rm -rf LiveOS/
root@debian:~/iso/centos/work# mksquashfs rootfs/ ./LiveOS/squashfs.img -comp xz -b 1048576 -Xdict-size 100%
root@debian:~/iso/centos/work# rm -rf rootfs/
root@debian:~/iso/centos/work# cd ../
root@debian:~/iso/centos# xorriso -as mkisofs \
    -o ~/iso/centos/Custom-CentOS.iso \
    -b isolinux/isolinux.bin \
    -c isolinux/boot.cat \
    -no-emul-boot \
    -boot-load-size 4 \
    -boot-info-table \
    -eltorito-alt-boot \
    -e images/efiboot.img \
    -no-emul-boot \
    -isohybrid-gpt-basdat \
    -V "ISOCDROM" \
    ~/iso/centos/work

