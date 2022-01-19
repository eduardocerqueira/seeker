#date: 2022-01-19T17:10:13Z
#url: https://api.github.com/gists/e9ef3c6c5ecb18a84bae28f4281847f1
#owner: https://api.github.com/users/caioau

#!/bin/bash

if [ $USER != 'root' ]; then
    echo "esse script deve rodar como root, rodeo novamente com sudo"
    exit 1
fi


echo "script para demonstrar como containers funcionam (namespaces, cgroups, chroot)"

# instala dependencias, caso nao tenha instaladas descomente
echo -e "instalando as dependencias necessarias (debootstrap cgroup-tools util-linux)\n"
sudo apt update && apt install debootstrap cgroup-tools util-linux

# cria a raiz de uma instalação debian na pasta debian11-rootfs
# descomente caso nao tenha feito ainda
if [ ! -d "./debian11-rootfs" ]; then
    echo -e "\n\ninstalando o debian 11 (bullseye) na pasta debian11-rootfs\n"
    debootstrap bullseye ./debian11-rootfs https://deb.debian.org/debian/
fi

echo "criando o cgroup: "

# cria um numero aleatorio entre 2000~3000 para ter cgroup unico
cgroup_name="cg_$(shuf -i 2000-3000 -n 1)"

cgcreate -g "cpu,cpuacct,memory,pids:$cgroup_name"

cgset -r cpu.shares=256 "$cgroup_name" # 0.25 cpu
cgset -r memory.limit_in_bytes=100M "$cgroup_name" # limite de 100MB RAM
cgset -r pids.max=100 "$cgroup_name" # no maximo 100 procesos simultaneos (forkbomb prevetion)

echo -e "\n\ncgroup criado, seu nome eh: $cgroup_name"
echo -e "iniciando o container, divirta-se\n"

# calma, muita coisa ao mesmo tempo: usa o cgroup que acabamos de criar (cgexec), 
#     cria namespaces novos (unshare), faz chroot e muda o hostname do container

cgexec -g "cpu,cpuacct,memory,pids:$cgroup_name" \
    unshare --fork --mount --uts --ipc --pid --mount-proc \
    chroot "./debian11-rootfs" \
    /bin/sh -c "/bin/mount -t proc proc /proc && hostname container && /bin/bash"
