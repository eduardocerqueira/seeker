#date: 2025-06-27T16:58:21Z
#url: https://api.github.com/gists/e8f5cd6f2c67d5459f3d8b7c4bd77f02
#owner: https://api.github.com/users/benana2113

#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# ===============================================
#         KONFIGURASI PENGGUNA (SESUAIKAN JIKA BERBEDA DARI SKRIP INSTALASI UTAMA)
# ===============================================
USERNAME="badjals" # Pastikan ini sama dengan USERNAME di skrip instalasi utama Anda

# ===============================================
#         POST-INSTALL PRE-CHECKS
# ===============================================

echo "---"
echo ">> Memulai skrip post-instalasi Arch Linux (MODE SUPER RINGAN + Kustomisasi)..."
echo "---"

# === CEK UNTUK PENGGUNA NON-ROOT ===
if [ "$(id -u)" -eq 0 ]; then
  echo "PERINGATAN: Skrip ini sebaiknya dijalankan sebagai pengguna biasa (misal: $USERNAME) dengan sudo."
  echo "Menjalankan sebagai root bisa menyebabkan masalah kepemilikan file atau perilaku tak terduga."
  echo "Keluar sekarang. Harap jalankan skrip ini dari akun pengguna Anda menggunakan: ./post_install.sh"
  exit 1
fi

# === CEK KONEKSI INTERNET ===
echo ">> Memeriksa koneksi internet..."
if ! ping -c 1 archlinux.org &> /dev/null; then
  echo "ERROR: Tidak ada koneksi internet. Pastikan Anda terhubung."
  exit 1
fi
echo "Koneksi internet OK."

# ===============================================
#         KONFIGURASI INTI
# ===============================================

# === 1. Instalasi AUR Helper (yay) ===
echo "---"
echo ">> Menginstal AUR helper: yay..."

if ! command -v git &> /dev/null; then
    echo "ERROR: Git tidak ditemukan. Instal Git terlebih dahulu: sudo pacman -S git"
    exit 1
fi

if ! pacman -Qs base-devel &> /dev/null; then
   echo "Menginstal base-devel..."
   sudo pacman -S --noconfirm base-devel
fi

cd /tmp
if [ -d "yay" ]; then rm -rf yay; fi
echo "Menduplikasi repositori yay..."
git clone https://aur.archlinux.org/yay.git
cd yay
echo "Membangun dan menginstal yay..."
makepkg -si --noconfirm
cd /tmp
rm -rf yay
echo "Yay berhasil diinstal."

# === 2. Instalasi libinput-gestures dan xdotool ===
echo "---"
echo ">> Menginstal libinput-gestures dan xdotool..."
yay -S --noconfirm libinput-gestures xdotool
echo "libinput-gestures dan xdotool berhasil diinstal."

# --- Konfigurasi libinput-gestures ---
echo "Mengatur konfigurasi libinput-gestures..."
sudo sh -c 'echo "gesture swipe up 3 xdotool key super" > /etc/libinput-gestures.conf'
sudo sh -c 'echo "gesture swipe down 3 xdotool key super+Shift+q" >> /etc/libinput-gestures.conf'
sudo sh -c 'echo "gesture swipe left 3 xdotool key super+Left" >> /etc/libinput-gestures.conf'
sudo sh -c 'echo "gesture swipe right 3 xdotool key super+Right" >> /etc/libinput-gestures.conf'

# Autostart libinput-gestures-daemon di .xinitrc
if ! grep -q "libinput-gestures-daemon &" "/home/$USERNAME/.xinitrc"; then
    # Jika baris exec i3 ada di .xinitrc, kita masukkan libinput-gestures di atasnya
    if grep -q "exec i3" "/home/$USERNAME/.xinitrc"; then
        sed -i "/exec i3/i libinput-gestures-daemon &" "/home/$USERNAME/.xinitrc"
    else
        echo "libinput-gestures-daemon &" >> "/home/$USERNAME/.xinitrc"
    fi
fi
chown "$USERNAME:$USERNAME" "/home/$USERNAME/.xinitrc"


# === 3. Instalasi nvidia-prime (untuk prime-run) ===
echo "---"
echo ">> Menginstal nvidia-prime (untuk utilitas prime-run)..."
yay -S --noconfirm nvidia-prime
echo "nvidia-prime berhasil diinstal. Anda sekarang dapat menggunakan 'prime-run <aplikasi>'."

# === 4. Instalasi Paket Audio (PENTING untuk suara) ===
echo "---"
echo ">> Menginstal paket audio (PipeWire) dan utilitas dasar..."
AUDIO_PACKAGES=(
    sof-firmware
    alsa-utils
    pipewire pipewire-alsa pipewire-pulse pipewire-jack wireplumber
)
echo "Daftar paket audio yang akan diinstal: ${AUDIO_PACKAGES[@]}"
yay -S --noconfirm "${AUDIO_PACKAGES[@]}"
echo "Instalasi paket audio selesai."

# === 5. Instalasi Paket Kustomisasi Desktop & Sistem Login ===
echo "---"
echo ">> Menginstal paket untuk kustomisasi desktop dan sistem login..."

DESKTOP_CUSTOM_PACKAGES=(
    feh             # Untuk wallpaper
    polybar         # Status bar (alternatif i3bar)
    picom           # Compositor untuk efek visual (transparansi, bayangan)
    lxappearance    # Untuk mengatur tema GTK
    qt5ct           # Untuk mengatur tema Qt5
    # qt6ct         # Opsional, jika Anda punya aplikasi Qt6
    # clipmenu      # Clipboard manager (sudah ada di skrip Anda)
    # nmtui         # NetworkManager Text User Interface (jny jika perlu GUI jaringan CLI)
)

echo "Daftar paket kustomisasi desktop yang akan diinstal: ${DESKTOP_CUSTOM_PACKAGES[@]}"
yay -S --noconfirm "${DESKTOP_CUSTOM_PACKAGES[@]}"

# Display Manager yang sangat ringan (ly)
echo "Menginstal Display Manager ringan: ly..."
yay -S --noconfirm ly
echo "Ly Display Manager berhasil diinstal."


echo "Instalasi paket kustomisasi selesai."


# === 6. Konfigurasi Firewall (UFW) ===
echo "---"
echo ">> Menginstal dan mengkonfigurasi Firewall (UFW)..."
sudo pacman -S --noconfirm ufw
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
# Jika Anda tahu akan membutuhkan SSH, buka portnya:
# sudo ufw allow ssh
sudo systemctl enable ufw.service
sudo systemctl start ufw.service
echo "Firewall (UFW) berhasil diinstal dan diaktifkan."

# === 7. Manajemen Daya Laptop (TLP) ===
echo "---"
echo ">> Menginstal dan mengaktifkan Manajemen Daya Laptop (TLP)..."
sudo pacman -S --noconfirm tlp tlp-rdw
sudo systemctl enable tlp.service
sudo systemctl enable tlp-sleep.service
sudo systemctl start tlp.service # Mulai layanan segera
echo "Manajemen Daya (TLP) berhasil diinstal dan diaktifkan."

# === 8. Konfigurasi Layanan Tambahan (Lanjutan) ===
echo "---"
echo ">> Mengaktifkan layanan PipeWire (User services)..."
systemctl --user enable pipewire pipewire-pulse wireplumber

# Aktifkan Display Manager (ly)
echo "Mengaktifkan layanan Display Manager (ly.service)..."
sudo systemctl enable ly.service
# CATATAN: Layanan ly akan berjalan setelah reboot.

# Contoh: Set Wallpaper Otomatis dengan feh (jika feh diinstal)
# Anda bisa mengganti '/path/to/your/wallpaper.jpg' dengan path gambar Anda.
# Jika Anda ingin menyimpan wallpaper di home directory, pastikan itu ada di sana.
WALLPAPER_PATH="/home/$USERNAME/wallpaper.jpg" # Ganti dengan path wallpaper default Anda
# Pastikan wallpaper ada sebelum mencoba mengaturnya!
# if [ -f "$WALLPAPER_PATH" ]; then
#     echo "feh --bg-fill $WALLPAPER_PATH &" >> "/home/$USERNAME/.xinitrc"
# else
#     echo "WARNING: Wallpaper default tidak ditemukan di $WALLPAPER_PATH. Harap setel manual."
# fi

# Tambahkan Picom ke autostart i3 (jika diinstal)
if ! grep -q "picom --experimental-backends &" "/home/$USERNAME/.xinitrc"; then
    # Jika baris exec i3 ada, masukkan picom di atasnya
    if grep -q "exec i3" "/home/$USERNAME/.xinitrc"; then
        sed -i "/exec i3/i picom --experimental-backends &" "/home/$USERNAME/.xinitrc"
    else
        echo "picom --experimental-backends &" >> "/home/$USERNAME/.xinitrc"
    fi
fi
chown "$USERNAME:$USERNAME" "/home/$USERNAME/.xinitrc"


# === 9. Finalisasi Hak Akses (untuk memastikan semuanya milik pengguna) ===
echo "---"
echo ">> Memastikan hak akses file-file konfigurasi pengguna sudah benar..."
chown -R "$USERNAME:$USERNAME" "/home/$USERNAME/"
echo "Hak akses pengguna telah dikoreksi."

echo "---"
echo "✅ Skrip post-instalasi (MODE SUPER RINGAN + Kustomisasi) selesai!"
echo "Sistem Anda sekarang sangat minimalis, aman, efisien daya, dan siap untuk digunakan dengan Display Manager dan kustomisasi awal."
echo "---"
echo "BEBERAPA LANGKAH PENTING BERIKUTNYA:"
echo "1. SEGERA UBAH PASSWORD DEFAULT ANDA (untuk user dan root)."
echo "   Ketik 'passwd' untuk user, dan 'sudo passwd root' untuk root."
echo "2. Untuk menjalankan aplikasi dengan GPU NVIDIA: 'prime-run <nama_aplikasi>'"
echo "3. Sekarang Anda akan melihat layar login grafis 'ly' setelah reboot. Login dan nikmati!"
echo "4. Kustomisasi lebih lanjut: ~/.config/i3/config, ~/.config/polybar/config, ~/.config/picom/picom.conf"
echo "5. Selalu perbarui sistem secara teratur: 'yay -Syu'"
echo "6. Instal aplikasi yang Anda butuhkan secara manual dengan 'yay -S <nama_paket>' (misal: pcmanfm, gimp, vlc, libreoffice)."
echo "---"
