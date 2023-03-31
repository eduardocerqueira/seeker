#date: 2023-03-31T17:05:02Z
#url: https://api.github.com/gists/9a48f07bc1f919cd89e5715409f7d359
#owner: https://api.github.com/users/tommybuado

#!/bin/sh
# create 10-thinkpad-x260.hook script
cat > /etc/kernel-hooks.d/10-thinkpad-x260.hook <<EOF
#!/bin/sh
KERNEL_FLAVOR=\$1

if [ "\$KERNEL_FLAVOR" = "lts" ]; then
	ALPINE_LINUX="/boot/efi/EFI/alpine"
	if [ ! -d "\$ALPINE_LINUX" ]; then
		mkdir -p \$ALPINE_LINUX
	fi

	cp -r /boot/initramfs-lts \$ALPINE_LINUX/initramfs-lts
	cp -r /boot/vmlinuz-lts \$ALPINE_LINUX/vmlinuz-lts
	cp -r /boot/intel-ucode.img \$ALPINE_LINUX/intel-ucode.img
fi

params="root=UUID=$(findmnt / -o UUID -n) rootflags=rw,subvolid=256 \\
rootfstype=btrfs modules=sd-mod,usb-storage,btrfs quiet console=tty0 console=ttyS0 \\
initrd=\EFI\alpine\intel-ucode.img initrd=\EFI\alpine\initramfs-lts"

efibootmgr | grep Boot0000 > /dev/null
if [ "\$?" -eq 0 ]; then
	efibootmgr -b 0000 -B -q
fi

echo "==> uefi: creating boot entry - ThinkPad x260 [Alpine Linux]"
efibootmgr -b 0000 -c -d /dev/sda -p 1 -L "ThinkPad x260 [Alpine Linux]" \\
	-l "\EFI\alpine\vmlinuz-lts" -u "\${params}" -q
EOF

# make 10-thinkpad-x260.hook script as executable
chmod 0755 /etc/kernel-hooks.d/10-thinkpad-x260.hook
