#date: 2025-11-28T17:13:53Z
#url: https://api.github.com/gists/f9b96cbfc65f5dce3bd48655e01da905
#owner: https://api.github.com/users/RezaAmbler

#!/bin/bash
# ============================================================================
#  Deliberately Insecure "Hellbox" for PEEK / Nmap / NSE / TLS testing
#
#  - Exposes a ton of services to the internet
#  - Weak TLS, self-signed short/weak cert
#  - Anonymous FTP with write
#  - Open Redis / Memcached
#  - Weak DB users
#  - Lax SSH / Samba / mail / IMAP configs
#
#  ONLY USE ON THROWAWAY LAB INSTANCES
# ============================================================================

set -u -o pipefail
export DEBIAN_FRONTEND=noninteractive

echo "=== [1] Installing packages ==="
apt update -y

# Keep it big but not insane; some may already be installed
apt install -y \
  nginx apache2 \
  vsftpd \
  dovecot-imapd dovecot-pop3d \
  postfix \
  mysql-server postgresql \
  openssh-server \
  samba \
  redis-server memcached \
  xrdp \
  inetutils-inetd tftpd-hpa ftp || \
  echo "[WARN] Some packages failed to install; continuing."

# --------------------------------------------------------------------------
#  Weak TLS cert (short-lived, weak key, SHA1)
# --------------------------------------------------------------------------
echo
echo "=== [2] Generating weak, self-signed TLS cert ==="
BADCERT_DIR="/root/badcert"
mkdir -p "$BADCERT_DIR"

openssl req -x509 -newkey rsa:1024 \
  -keyout "$BADCERT_DIR/weak.key" \
  -out "$BADCERT_DIR/weak.crt" \
  -sha1 -days 90 \
  -nodes \
  -subj "/CN=hellbox.local" || \
  echo "[WARN] OpenSSL cert generation failed."

chmod 600 "$BADCERT_DIR/weak.key"

# --------------------------------------------------------------------------
#  Apache: weak TLS, directory listing on, self-signed cert
# --------------------------------------------------------------------------
echo
echo "=== [3] Apache HTTPS with weak TLS and directory listing ==="

a2enmod ssl >/dev/null 2>&1 || true
a2enmod headers >/dev/null 2>&1 || true

cat >/etc/apache2/sites-available/000-default-ssl.conf <<'EOF'
<VirtualHost *:443>
    ServerName hellbox.local

    SSLEngine on
    SSLCertificateFile /root/badcert/weak.crt
    SSLCertificateKeyFile /root/badcert/weak.key

    # Allow older TLS versions, disable TLS1.3
    SSLProtocol all -TLSv1.3

    # Intentionally bad cipher list
    SSLCipherSuite "ALL:!SECURE:!HIGH"
    SSLHonorCipherOrder on

    DocumentRoot /var/www/html
    <Directory /var/www/html>
        Options Indexes FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>

    # Clickjacking, XSS protections disabled (for testing)
    Header always unset X-Frame-Options
    Header always unset X-Content-Type-Options
    Header always unset X-XSS-Protection
</VirtualHost>
EOF

a2ensite 000-default-ssl.conf >/dev/null 2>&1 || true
systemctl enable apache2 >/dev/null 2>&1 || true
systemctl restart apache2 || echo "[WARN] Apache restart failed."

# --------------------------------------------------------------------------
#  Nginx: weak TLS, autoindex, same cert
# --------------------------------------------------------------------------
echo
echo "=== [4] Nginx HTTPS with weak TLS and autoindex ==="

cat >/etc/nginx/sites-available/default <<'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    root /var/www/html;
    index index.html index.htm;

    # Directory listing
    location / {
        autoindex on;
    }
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name _;

    ssl_certificate     /root/badcert/weak.crt;
    ssl_certificate_key /root/badcert/weak.key;

    # Allow old protocols
    ssl_protocols TLSv1 TLSv1.1 TLSv1.2;

    # Very permissive ciphers
    ssl_ciphers ALL:!SECURE:!HIGH;
    ssl_prefer_server_ciphers on;

    root /var/www/html;
    index index.html index.htm;

    location / {
        autoindex on;
    }
}
EOF

nginx -t && systemctl enable nginx >/dev/null 2>&1 && systemctl restart nginx || \
  echo "[WARN] Nginx config test or restart failed."

# --------------------------------------------------------------------------
#  Postfix: weak-ish TLS; still avoid full open relay
# --------------------------------------------------------------------------
echo
echo "=== [5] Postfix with weak TLS (no relay lock-down hardening) ==="

postconf -e "smtpd_tls_cert_file=/root/badcert/weak.crt"
postconf -e "smtpd_tls_key_file=/root/badcert/weak.key"
postconf -e "smtpd_tls_security_level=may"
postconf -e "smtpd_tls_mandatory_protocols=!TLSv1.3"
postconf -e "smtp_tls_mandatory_protocols=!TLSv1.3"
postconf -e "smtpd_tls_mandatory_ciphers=export"
postconf -e "smtp_use_tls=yes"
# Allow plain auth on non-TLS (bad)
postconf -e "smtpd_tls_auth_only=no"

systemctl enable postfix >/dev/null 2>&1 || true
systemctl restart postfix || echo "[WARN] Postfix restart failed."

# --------------------------------------------------------------------------
#  Dovecot: allow plaintext auth + weak TLS
# --------------------------------------------------------------------------
echo
echo "=== [6] Dovecot (IMAP/POP3) with plaintext auth and weak TLS ==="

DOVECOT_SSL_CONF="/etc/dovecot/conf.d/10-ssl.conf"
if [ -f "$DOVECOT_SSL_CONF" ]; then
  cp "$DOVECOT_SSL_CONF" "${DOVECOT_SSL_CONF}.bak.$(date +%s)" || true
fi

cat >"$DOVECOT_SSL_CONF" <<'EOF'
ssl = yes
ssl_cert = </root/badcert/weak.crt
ssl_key  = </root/badcert/weak.key

# Allow old protocols; disable TLS1.3
ssl_protocols = !TLSv1.3

# Intentionally weak cipher list
ssl_cipher_list = ALL:!SECURE:!HIGH
EOF

# Allow plaintext auth (very bad)
AUTH_CONF="/etc/dovecot/conf.d/10-auth.conf"
if [ -f "$AUTH_CONF" ]; then
  sed -i 's/^#\?disable_plaintext_auth.*/disable_plaintext_auth = no/' "$AUTH_CONF" || true
fi

systemctl enable dovecot >/dev/null 2>&1 || true
systemctl restart dovecot || echo "[WARN] Dovecot restart failed."

# --------------------------------------------------------------------------
#  vsftpd: anonymous writeable FTP
# --------------------------------------------------------------------------
echo
echo "=== [7] vsftpd: anonymous writable FTP ==="

FTP_ROOT="/srv/ftp"
mkdir -p "$FTP_ROOT"
chmod 777 "$FTP_ROOT"

cat >/etc/vsftpd.conf <<'EOF'
listen=YES
listen_ipv6=NO
anonymous_enable=YES
local_enable=YES
write_enable=YES
anon_upload_enable=YES
anon_mkdir_write_enable=YES
anon_other_write_enable=YES
anon_root=/srv/ftp
no_anon_password= "**********"
xferlog_std_format=YES
pasv_min_port=30000
pasv_max_port=30010
EOF

systemctl enable vsftpd >/dev/null 2>&1 || true
systemctl restart vsftpd || echo "[WARN] vsftpd restart failed."

# --------------------------------------------------------------------------
#  SSH: "**********"
# --------------------------------------------------------------------------
echo
echo "=== [8] SSH: "**********"

SSHD_CFG="/etc/ssh/sshd_config"
cp "$SSHD_CFG" "${SSHD_CFG}.bak.$(date +%s)" || true

sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' "$SSHD_CFG" || true
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' "$SSHD_CFG" || true

systemctl enable ssh >/dev/null 2>&1 || true
systemctl restart ssh || echo "[WARN] SSH restart failed."

# --------------------------------------------------------------------------
#  MySQL: listen on 0.0.0.0; weak test user
# --------------------------------------------------------------------------
echo
echo "=== [9] MySQL: open bind + weak user ==="

MYCNF="/etc/mysql/mysql.conf.d/mysqld.cnf"
if [ -f "$MYCNF" ]; then
  sed -i 's/^\s*bind-address.*/bind-address = 0.0.0.0/' "$MYCNF" || \
    echo "[WARN] Could not update MySQL bind-address."
fi

systemctl enable mysql >/dev/null 2>&1 || true
systemctl restart mysql || echo "[WARN] MySQL restart failed."

mysql -u root <<'EOF' || echo "[WARN] MySQL user creation failed."
CREATE USER IF NOT EXISTS 'test'@'%' IDENTIFIED BY 'test123';
GRANT ALL PRIVILEGES ON *.* TO 'test'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
EOF

# --------------------------------------------------------------------------
#  PostgreSQL: listen on 0.0.0.0; weak test user/db
# --------------------------------------------------------------------------
echo
echo "=== [10] PostgreSQL: open listen + weak user ==="

PG_VER=$(psql -V 2>/dev/null | awk '{print $3}' | cut -d. -f1 || echo "16")
PG_CONF="/etc/postgresql/${PG_VER}/main/postgresql.conf"
PG_HBA="/etc/postgresql/${PG_VER}/main/pg_hba.conf"

if [ -f "$PG_CONF" ]; then
  sed -i "s/^#\?listen_addresses.*/listen_addresses = '*'/" "$PG_CONF" || true
fi

if [ -f "$PG_HBA" ]; then
  cp "$PG_HBA" "${PG_HBA}.bak.$(date +%s)" || true
  echo "host    all             all             0.0.0.0/0               md5" >> "$PG_HBA"
fi

systemctl enable postgresql >/dev/null 2>&1 || true
systemctl restart postgresql || echo "[WARN] PostgreSQL restart failed."

sudo -u postgres psql <<'EOF' || echo "[WARN] PostgreSQL user/db creation failed."
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'testuser') THEN
      CREATE ROLE testuser LOGIN PASSWORD 'test123';
   END IF;
END$$;
CREATE DATABASE testdb OWNER testuser;
EOF

# --------------------------------------------------------------------------
#  Redis: bind 0.0.0.0, protected-mode no
# --------------------------------------------------------------------------
echo
echo "=== [11] Redis: open to the world, no protected-mode ==="

REDIS_CONF="/etc/redis/redis.conf"
if [ -f "$REDIS_CONF" ]; then
  cp "$REDIS_CONF" "${REDIS_CONF}.bak.$(date +%s)" || true
  sed -i 's/^bind .*/bind 0.0.0.0/' "$REDIS_CONF" || true
  sed -i 's/^protected-mode .*/protected-mode no/' "$REDIS_CONF" || true
fi

systemctl enable redis-server >/dev/null 2>&1 || true
systemctl restart redis-server || echo "[WARN] Redis restart failed."

# --------------------------------------------------------------------------
#  Memcached: listen on 0.0.0.0
# --------------------------------------------------------------------------
echo
echo "=== [12] Memcached: open to the world ==="

MEMCACHED_CONF="/etc/memcached.conf"
if [ -f "$MEMCACHED_CONF" ]; then
  cp "$MEMCACHED_CONF" "${MEMCACHED_CONF}.bak.$(date +%s)" || true
  sed -i 's/^-l .*/-l 0.0.0.0/' "$MEMCACHED_CONF" || true
fi

systemctl enable memcached >/dev/null 2>&1 || true
systemctl restart memcached || echo "[WARN] Memcached restart failed."

# --------------------------------------------------------------------------
#  Samba: guest writable share
# --------------------------------------------------------------------------
echo
echo "=== [13] Samba: guest writable share ==="

SMB_CONF="/etc/samba/smb.conf"
if [ -f "$SMB_CONF" ]; then
  cp "$SMB_CONF" "${SMB_CONF}.bak.$(date +%s)" || true
fi

SHARE_DIR="/srv/samba/public"
mkdir -p "$SHARE_DIR"
chmod 777 "$SHARE_DIR"

cat >"$SMB_CONF" <<'EOF'
[global]
   workgroup = WORKGROUP
   server string = Hellbox Samba Server
   map to guest = Bad User
   dns proxy = no

[public]
   path = /srv/samba/public
   browseable = yes
   writable = yes
   guest ok = yes
   read only = no
EOF

systemctl enable smbd nmbd >/dev/null 2>&1 || true
systemctl restart smbd nmbd || echo "[WARN] Samba restart failed."

# --------------------------------------------------------------------------
#  XRDP: default is enough (just ensure running)
# --------------------------------------------------------------------------
echo
echo "=== [14] XRDP: ensure running ==="

systemctl enable xrdp >/dev/null 2>&1 || true
systemctl restart xrdp || echo "[WARN] XRDP restart failed."

# --------------------------------------------------------------------------
#  TFTP: open, world-writable
# --------------------------------------------------------------------------
echo
echo "=== [15] TFTP: open, world-writable ==="

TFTP_ROOT="/srv/tftp"
mkdir -p "$TFTP_ROOT"
chmod 777 "$TFTP_ROOT"

cat >/etc/default/tftpd-hpa <<EOF
TFTP_USERNAME="tftp"
TFTP_DIRECTORY="$TFTP_ROOT"
TFTP_ADDRESS="0.0.0.0:69"
TFTP_OPTIONS="--secure --create"
EOF

systemctl enable inetutils-inetd tftpd-hpa >/dev/null 2>&1 || true
systemctl restart inetutils-inetd tftpd-hpa || echo "[WARN] TFTP/inetd restart failed."

# --------------------------------------------------------------------------
#  Summary
# --------------------------------------------------------------------------
echo
echo "=== [16] Summary for scanners ==="
PUBIP=$(curl -s ifconfig.me || echo "<public-ip>")

cat <<EOF

Hellbox public IP: $PUBIP

Deliberately exposed / misconfigured services:

  HTTP/HTTPS:
    - Apache:    http://$PUBIP, https://$PUBIP (weak TLS, dir listing)
    - Nginx:     http://$PUBIP, https://$PUBIP (weak TLS, autoindex)

  Mail / IMAP / POP:
    - Postfix:   $PUBIP:25, 587 (weak TLS, plain auth allowed)
    - Dovecot:   $PUBIP:143/993 (IMAP), 110/995 (POP; plaintext allowed)

  Databases:
    - MySQL:     $PUBIP:3306 (user:test / pass:test123, open bind)
    - Postgres:  $PUBIP:5432 (user:testuser / pass:test123, db:testdb, open listen)

  Auth / Shell:
    - SSH: "**********":22 (root login + password auth enabled)

  File services:
    - FTP:       $PUBIP:21 (anonymous writable /srv/ftp)
    - Samba:     $PUBIP:445 (share: public, guest ok, writable)
    - TFTP:      $PUBIP:69 (world-writable /srv/tftp)

  Caches:
    - Redis:     $PUBIP:6379 (bind 0.0.0.0, protected-mode no)
    - Memcached: $PUBIP:11211 (listen 0.0.0.0)

  RDP:
    - XRDP:      $PUBIP:3389

TLS cert:
  - /root/badcert/weak.crt (RSA 1024, SHA1, short validity)

This box is intentionally terrible. Perfect for PEEK / Nmap / NSE / sslscan testing.
EOF

echo "=== Hellbox setup complete ==="echo "=== Hellbox setup complete ==="