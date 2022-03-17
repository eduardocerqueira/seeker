#date: 2022-03-17T16:54:41Z
#url: https://api.github.com/gists/2bf86b66ca4dc914a5aecaddf9f04ec1
#owner: https://api.github.com/users/rwsu

virt-install --connect qemu:///system -n control1 -r 17000 --vcpus 6 --cdrom ./output/fleeting.iso --disk pool=default,size=60 --boot hd,cdrom --os-variant=fedora-coreos-stable --network network=default,mac=52:54:01:aa:aa:a1 --wait=-1 &
sleep 2
virt-install --connect qemu:///system -n control2 -r 17000 --vcpus 6 --cdrom ./output/fleeting.iso --disk pool=default,size=60 --boot hd,cdrom --os-variant=fedora-coreos-stable --network network=default,mac=52:54:01:bb:bb:b1 --wait=-1 &
sleep 2
virt-install --connect qemu:///system -n control3 -r 17000 --vcpus 6 --cdrom ./output/fleeting.iso --disk pool=default,size=60 --boot hd,cdrom --os-variant=fedora-coreos-stable --network network=default,mac=52:54:01:cc:cc:c1 --wait=-1 &