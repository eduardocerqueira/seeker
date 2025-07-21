#date: 2025-07-21T16:42:17Z
#url: https://api.github.com/gists/9eb402150a395f25faaba67ad5e7d978
#owner: https://api.github.com/users/WandersonMatriX

#!/bin/bash
set -e

# === CONFIGURAÇÃO PERSONALIZADA ===
USERNAME="wanderson"
PASSWORD= "**********"

# === DISCO E PARTIÇÕES ===
PART_ROOT="/dev/sda5"   # <-- 200GB criados no espaço não alocado
PART_EFI="/dev/sda1"    # <-- Partição EFI existente do Windows

# === 1. Sincronizar o relógio ===
timedatectl set-ntp true

# === 2. Formatar partição ROOT ===
mkfs.ext4 $PART_ROOT

# === 3. Montar partições ===
mount $PART_ROOT /mnt
mount --mkdir $PART_EFI /mnt/boot/efi

# === 4. Instalar sistema base e GNOME ===
pacstrap /mnt base linux linux-firmware networkmanager sudo grub efibootmgr gnome gnome-tweaks gdm bash-completion

# === 5. Gerar fstab ===
genfstab -U /mnt >> /mnt/etc/fstab

# === 6. Configuração dentro do sistema ===
arch-chroot /mnt /bin/bash <<EOF
# === Timezone e locale ===
ln -sf /usr/share/zoneinfo/America/Fortaleza /etc/localtime
hwclock --systohc
echo "pt_BR.UTF-8 UTF-8" >> /etc/locale.gen
locale-gen
echo "LANG=pt_BR.UTF-8" > /etc/locale.conf
echo "$USERNAME-pc" > /etc/hostname

# === Hosts ===
cat > /etc/hosts <<EOL
127.0.0.1   localhost
::1         localhost
127.0.1.1   $USERNAME-pc.localdomain $USERNAME-pc
EOL

# === Criar usuário ===
useradd -m -G wheel $USERNAME
echo "$USERNAME: "**********"
echo "root: "**********"

# === Permitir sudo ===
echo "%wheel ALL=(ALL) ALL" >> /etc/sudoers

# === Ativar serviços ===
systemctl enable NetworkManager
systemctl enable gdm

# === Instalar e configurar GRUB ===
grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=Arch
grub-mkconfig -o /boot/grub/grub.cfg
EOF

echo -e "\n✅ Instalação concluída. Agora você pode reiniciar!"

iniciar!"

