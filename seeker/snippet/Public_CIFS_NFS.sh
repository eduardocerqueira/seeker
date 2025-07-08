#date: 2025-07-08T16:36:51Z
#url: https://api.github.com/gists/06376403eeb9f10d103cbb4d4350ceb2
#owner: https://api.github.com/users/casjay

#!/bin/sh
# ┌────────────────────────────────────────────────────┐
# │     🚀 NFS + CIFS Public Server Installer (Debian) │
# └────────────────────────────────────────────────────┘

# ✨ POSIX-compliant, emoji-balanced, zero-noise unless error

# 🎨 Colors
RED="\\033[1;31m"
GRN="\\033[1;32m"
YEL="\\033[1;33m"
BLU="\\033[1;34m"
MAG="\\033[1;35m"
CYN="\\033[1;36m"
RST="\\033[0m"

# 🎤 Output Helpers
say() { printf "${CYN}💬 %s${RST}\\n" "$*" >&3; }
status() { printf "${GRN}✅ %s${RST}\\n" "$*" >&3; }
warn() { printf "${YEL}⚠️  %s${RST}\\n" "$*" >&3; }
fail() {
  printf "${RED}💥 ERROR on line $1 [exit $2]${RST}\\n" >&2
  exit "$2"
}

trap 'fail "$LINENO" "$?"' ERR
set -e
exec 3>&1

# 📦 Config
SHARE_BASE="/var/ftp/pub"
EXT_IP=`curl -s http://ifcfg.us/ip 2>/dev/null || hostname -I | awk '{print $1}'`
HOSTNAME_FQDN=`hostname -f`
HOSTNAME_SHORT=`echo "$HOSTNAME_FQDN" | cut -d. -f1`
ADMIN_USER="administrator"

say "🌐 Detected external IP: $EXT_IP"
say "🖥️ Hostname: $HOSTNAME_FQDN"

# 📥 Install Dependencies
say "📦 Installing NFS and Samba..."
apt update >/dev/null 2>&1
apt install -y nfs-kernel-server samba >/dev/null 2>&1
status "Dependencies installed"

# 📁 Create Shares
say "📁 Setting up public directories..."
mkdir -p "$SHARE_BASE/ISOs" "$SHARE_BASE/mirrors"
chown -R ftp:root "$SHARE_BASE"
chmod -R 755 "$SHARE_BASE"
status "Directories created"

# 📡 NFS Configuration
say "🔧 Configuring NFS exports..."
cat > /etc/exports <<EOF
$SHARE_BASE/ISOs     *(ro,sync,no_subtree_check,no_root_squash)
$SHARE_BASE/mirrors  *(ro,sync,no_subtree_check,no_root_squash)
$SHARE_BASE          *(ro,sync,no_subtree_check,no_root_squash)
EOF

exportfs -a >/dev/null 2>&1
systemctl enable nfs-server >/dev/null 2>&1
systemctl start nfs-server >/dev/null 2>&1
status "NFS server running"

# 🛜 Samba Configuration
say "🛠️ Setting up Samba..."

[ -f /etc/samba/smb.conf ] && mv /etc/samba/smb.conf /etc/samba/smb.conf.bak && warn "Backed up old smb.conf"

cat > /etc/samba/smb.conf <<EOF
[global]
   server string = 🌍 Public CIFS Server
   workgroup = WORKGROUP
   netbios name = $HOSTNAME_SHORT
   security = user
   map to guest = Bad User
   guest account = ftp
   unix password sync = "**********"
   log file = /var/log/samba/log.%m
   max log size = 1000
   server role = standalone server
   passdb backend = tdbsam
   username map = /etc/samba/smbusers

[ISOs]
   path = $SHARE_BASE/ISOs
   public = yes
   browsable = yes
   guest ok = yes
   read only = yes
   force user = ftp
   force group = root

[Mirrors]
   path = $SHARE_BASE/mirrors
   public = yes
   browsable = yes
   guest ok = yes
   read only = yes
   force user = ftp
   force group = root

[FTP]
   path = $SHARE_BASE
   public = yes
   browsable = yes
   guest ok = yes
   read only = yes
   force user = ftp
   force group = root
EOF

echo "$ADMIN_USER = root" > /etc/samba/smbusers
echo "root" | smbpasswd -s -a root >/dev/null 2>&1
pdbedit -a -u root >/dev/null 2>&1
systemctl enable smbd nmbd >/dev/null 2>&1
systemctl start smbd nmbd >/dev/null 2>&1
status "Samba service running"

# 🔐 Optional: UFW
if command -v ufw >/dev/null 2>&1; then
  say "🛡️ Allowing NFS and Samba through UFW..."
  ufw allow from any to any port nfs >/dev/null 2>&1
  ufw allow samba >/dev/null 2>&1
  status "Firewall rules applied"
fi

# 🎉 Summary
printf "\n${MAG}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}\n" >&3
printf "${BLU}🎉 NFS + CIFS Server Setup Complete${RST}\n" >&3
printf "${MAG}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}\n" >&3
printf "${CYN}🌍 External IP:  ${RST}%s\n" "$EXT_IP" >&3
printf "${CYN}📡 NFS Share:    ${RST}nfs://$EXT_IP/var/ftp/pub\n" >&3
printf "${CYN}🛜 CIFS Share:   ${RST}smb://$EXT_IP/\n" >&3
printf "${MAG}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}\n" >&3

cat <<EOF >&3

📦 NFS Mounting:
  sudo mount -t nfs $EXT_IP:/var/ftp/pub/ISOs /mnt/isos
  sudo mount -t nfs $EXT_IP:/var/ftp/pub/mirrors /mnt/mirrors
  sudo mount -t nfs $EXT_IP:/var/ftp/pub /mnt/ftp

🛜 CIFS Guest Mounts (Read-Only):
  sudo mount -t cifs //$EXT_IP/ISOs /mnt/isos -o guest,ro
  sudo mount -t cifs //$EXT_IP/mirrors /mnt/mirrors -o guest,ro
  sudo mount -t cifs //$EXT_IP/ftp /mnt/ftp -o guest,ro

🔑 CIFS Admin Access (Read-Write):
  sudo mount -t cifs //$EXT_IP/ISOs /mnt/isos -o username= "**********"=<root-password>,rw

EOF

exit 0
