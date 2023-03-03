#date: 2023-03-03T16:47:53Z
#url: https://api.github.com/gists/a4b3e611233748cfdc282ea7de1d4c3e
#owner: https://api.github.com/users/iruwl

## reff: https://gist.github.com/smoser/6066204


## download ubuntu cloud image
$ wget https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img

## install requirement
$ sudo apt-get install virtualbox qemu-utils genisoimage cloud-utils

## convert cloud image to raw
$ qemu-img convert -O raw jammy-server-cloudimg-amd64.img jammy-cloudimg.raw

## convert raw to virtualbox 'vdi' format
$ vboxmanage convertfromraw jammy-cloudimg.raw jammy-cloudimg-disk.vdi

## create user-data file and a iso file with that user-data on it
$ cat > my-user-data <<EOF
#cloud-config
password: "**********"
chpasswd: { expire: False }
ssh_pwauth: True
EOF
$ cloud-localds my-seed.iso my-user-data

## create a virtual machine
$ vboxmanage createvm --name "jammy-nocloud" --register
$ vboxmanage modifyvm "jammy-nocloud" \
   --memory 1024 --boot1 disk --acpi on \
   --nic1 nat --natpf1 "guestssh,tcp,,2222,,22"
$ vboxmanage storagectl "jammy-nocloud" --name "IDE_0" --add ide
$ vboxmanage storageattach "jammy-nocloud" \
    --storagectl "IDE_0" --port 0 --device 0 \
    --type hdd --medium "jammy-cloudimg-disk.vdi"
$ vboxmanage storageattach "jammy-nocloud" \
    --storagectl "IDE_0" --port 1 --device 0 \
    --type dvddrive --medium "my-seed.iso"

## start up the VM
$ vboxheadless --startvm "jammy-nocloud"

## Now, you should be able to connect to the VM using ssh.
## After the system boots, you can ssh in with 'ubuntu:passw0rd'
## command: ssh -p 2222 ubuntu@localhost

## power off the VM
$ vboxmanage controlvm "jammy-nocloud" poweroff

## delete vm
$ vboxmanage storageattach "jammy-nocloud" \
   --storagectl "IDE_0" --port 0 --device 0 --medium none
$ vboxmanage storageattach "jammy-nocloud" \
   --storagectl "IDE_0" --port 1 --device 0 --medium none
$ vboxmanage closemedium dvd "my-seed.iso"
$ vboxmanage closemedium disk "jammy-cloudimg-disk.vdi"
$ vboxmanage unregistervm "jammy-nocloud" --delete
