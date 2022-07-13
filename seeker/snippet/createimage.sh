#date: 2022-07-13T17:12:24Z
#url: https://api.github.com/gists/91e8a03e1cc36b02dd3c30c7fafe67c7
#owner: https://api.github.com/users/faoziaziz

mkdir u_aarch64
cd u_aarch64

wget -O ubuntu-16.04.7-server-arm64.iso http://cdimage.ubuntu.com/ubuntu/releases/16.04.7/release/ubuntu-16.04.7-server-arm64.iso&& \
wget https://releases.linaro.org/components/kernel/uefi-linaro/latest/release/qemu64/QEMU_EFI.fd && \
cp QEMU_EFI.fd flash0.img && \
truncate -s 64M flash0.img&& \
truncate -s 64M flash1.img


# generate image

qemu-img create -f qcow2 xenial.qcow2 16G&& \
 qemu-system-aarch64 -M virt -cpu cortex-a53 -m 4096 \
-drive if=pflash,format=raw,file=flash0.img,readonly \
-drive if=pflash,format=raw,file=flash1.img \
-drive if=none,file=xenial.qcow2,format=qcow2,id=hd \
-device virtio-blk-device,drive=hd \
-netdev type=user,id=mynet \
-device virtio-net-device,netdev=mynet \
-nographic -no-reboot \
-device virtio-scsi \
-drive if=none,id=cd,file=ubuntu-16.04.7-server-arm64.iso \
-device scsi-cd,drive=cd

# load image 
qemu-system-aarch64 -M virt -cpu cortex-a53 -m 4096 \
-drive if=pflash,format=raw,file=flash0.img,readonly \
-drive if=pflash,format=raw,file=flash1.img \
-drive if=none,file=xenial.qcow2,format=qcow2,id=hd \
-device virtio-blk-device,drive=hd \
-netdev type=user,id=mynet \
-device virtio-net-device,netdev=mynet \
-nographic

