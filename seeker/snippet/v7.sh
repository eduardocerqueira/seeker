#date: 2025-06-30T17:00:57Z
#url: https://api.github.com/gists/27081a8e13a611bb9150c0214a2759e6
#owner: https://api.github.com/users/Fantasycgc

#!/bin/bash

# === CONFIG ===
ERP_USER="erpnext"
ERP_PASS="erpnext"
SITE_NAME="erp.local"
SITE_PORT="9996"
ADMIN_PASS="admin123"
DB_ROOT_PASS="root"
USE_SSL=false # true nếu có domain
DOMAIN=""     # nhập domain nếu dùng SSL

### === Function Helpers === ###
run_as_erp() {
    sudo -H -u $ERP_USER bash -c "export PATH=\$HOME/.local/bin:\$PATH && $1"
}
### --- BẬT SYSTEMD --- ###
echo "🔧 Bật systemd trong WSL..."
sudo tee /etc/wsl.conf > /dev/null <<EOF
[boot]
systemd=true
EOF
echo "✅ Đã bật systemd trong /etc/wsl.conf"
echo "👉 Vui lòng chạy: wsl --shutdown từ PowerShell và mở lại WSL sau bước này."
sleep 5

### --- KIỂM TRA SYSTEMD --- ###
if [ "$(ps -p 1 -o comm=)" != "systemd" ]; then
  echo "❌ systemd chưa hoạt động. Vui lòng chạy: wsl --shutdown rồi mở lại WSL."
  exit 1
fi

### === 1. Cập nhật môi trường ===
echo "[1/9] 🧼 Updating system..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl python3-dev python3-pip python3-venv \
  mariadb-server redis-server xvfb libfontconfig wkhtmltopdf \
  libmysqlclient-dev libffi-dev libssl-dev libjpeg-dev libxrender1 supervisor nginx
  
 ### === Redis Multi-Port for ERPNext ===
echo "⚙️ Cấu hình Redis đa cổng cho ERPNext..."

# Tạo thư mục chứa cấu hình riêng cho Redis từng dịch vụ
sudo mkdir -p /etc/redis-erp

# Tạo và chỉnh sửa cấu hình redis cho từng dịch vụ (queue, socketio, cache)
for name in queue socketio cache; do
  case $name in
    queue) port=11000 ;;
    socketio) port=12000 ;;
    cache) port=13000 ;;
  esac

  conf="/etc/redis-erp/redis-$name.conf"
  sudo cp /etc/redis/redis.conf "$conf"
  sudo sed -i "s/^port .*/port $port/" "$conf"
  sudo sed -i "/^port/ a pidfile /var/run/redis-$name.pid" "$conf"
done

# Tạo file systemd service cho từng redis instance
for name in queue socketio cache; do
  case $name in
    queue) port=11000 ;;
    socketio) port=12000 ;;
    cache) port=13000 ;;
  esac

  cat <<EOF | sudo tee /etc/systemd/system/redis-$name.service
[Unit]
Description=Redis $name (ERPNext)
After=network.target

[Service]
ExecStart=/usr/bin/redis-server /etc/redis-erp/redis-$name.conf
ExecStop=/usr/bin/redis-cli -p $port shutdown
Restart=always
User=redis
Group=redis

[Install]
WantedBy=multi-user.target
EOF
done

echo "🛠 Đang cập nhật common_site_config.json với các cổng Redis chuẩn..."

CONFIG_FILE="/home/erpnext/erpnext15/sites/common_site_config.json"

# Kiểm tra file tồn tại
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Không tìm thấy file $CONFIG_FILE"
  exit 1
fi

# Sử dụng jq nếu có, nếu không dùng sed thay thế
if command -v jq &> /dev/null; then
  echo "✅ Dùng jq để chỉnh file JSON an toàn..."
  tmp=$(mktemp)
  jq '.redis_queue = "redis://127.0.0.1:11000" |
      .redis_socketio = "redis://127.0.0.1:12000" |
      .redis_cache = "redis://127.0.0.1:13000"' "$CONFIG_FILE" > "$tmp" && sudo mv "$tmp" "$CONFIG_FILE"
else
  echo "⚠️ Không có jq, đang dùng sed để chỉnh thủ công..."
  sudo sed -i 's#"redis_queue": *"[^"]*"#"redis_queue": "redis://127.0.0.1:11000"#' "$CONFIG_FILE"
  sudo sed -i 's#"redis_socketio": *"[^"]*"#"redis_socketio": "redis://127.0.0.1:12000"#' "$CONFIG_FILE"
  sudo sed -i 's#"redis_cache": *"[^"]*"#"redis_cache": "redis://127.0.0.1:13000"#' "$CONFIG_FILE"
fi

echo "✅ Đã cập nhật xong common_site_config.json:"
grep redis_ "$CONFIG_FILE"

# Reload systemd và khởi động dịch vụ redis riêng
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable redis-queue redis-socketio redis-cache
sudo systemctl start redis-queue redis-socketio redis-cache

### === 2. Tạo user erpnext (nếu chưa tồn tại) ===
if id "$ERP_USER" &>/dev/null; then
    echo "👤 User $ERP_USER đã tồn tại."
else
    echo "👤 Tạo user $ERP_USER..."
    sudo useradd -m -s /bin/bash "$ERP_USER"
    echo "$ERP_USER:$ERP_PASS" | sudo chpasswd
    sudo usermod -aG sudo $ERP_USER
fi

### === 3. Cài Node.js 18 từ NodeSource ===
echo "[2/9] 🧩 Installing Node.js 18.x..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

### === 4. Cài yarn, bench, pip ===
echo "[3/9] 🧱 Installing yarn & bench..."
sudo -H -u $ERP_USER bash -c "
  sudo npm install -g yarn
    pip3 install --user frappe-bench
"

### === 5. Tạo bench ===
echo "[4/9] 🏗 Creating bench erpnext15..."
sudo -H -u $ERP_USER bash -c "cd /home/$ERP_USER && export PATH=\$HOME/.local/bin:\$PATH && bench init erpnext15 --skip-redis-config-generation --frappe-branch version-15"


cd /home/$ERP_USER/erpnext15 || { echo "❌ Bench init failed"; exit 1; }

### === 6. Thêm ERPNext app ===
echo "[5/9] ⬇️ Getting erpnext app..."
run_as_erp "cd /home/$ERP_USER/erpnext15 && bench get-app --branch version-15 erpnext"

### === 7. Cài site và khởi tạo db ===
echo "[6/9] 🌐 Creating site $SITE_NAME..."
echo "[6/9] 🔐 Cấu hình MariaDB..."
sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED BY '$DB_ROOT_PASS'; FLUSH PRIVILEGES;" || { echo "❌ Không thể cập nhật mật khẩu root MariaDB"; exit 1; }

sudo mysql -u root -p"$DB_ROOT_PASS" -e "CREATE USER IF NOT EXISTS '$ERP_USER'@'localhost' IDENTIFIED BY '$ERP_PASS'; GRANT ALL PRIVILEGES ON *.* TO '$ERP_USER'@'localhost' WITH GRANT OPTION; FLUSH PRIVILEGES;"


run_as_erp "cd /home/$ERP_USER/erpnext15 && bench new-site $SITE_NAME --db-name $SITE_NAME --mariadb-root-password $DB_ROOT_PASS --admin-password $ADMIN_PASS --install-app erpnext"


### === 8. Cài app vào site ===
run_as_erp "cd /home/$ERP_USER/erpnext15 && bench --site $SITE_NAME install-app erpnext"

### === 9. Production setup (nginx, supervisor) ===
echo "[7/9] 🚀 Setup production..."
run_as_erp "cd /home/$ERP_USER/erpnext15 && bench setup production $ERP_USER"

### === 10. Thay đổi port nginx (nếu cần) ===
echo "[8/9] 🛠 Changing nginx port to $SITE_PORT..."
NGINX_CONF="/etc/nginx/sites-available/$SITE_NAME"
if [ -f "$NGINX_CONF" ]; then
    sudo sed -i "s/listen 80;/listen $SITE_PORT;/" "$NGINX_CONF"
    sudo ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/
    sudo nginx -t && sudo systemctl reload nginx
else
    echo "⚠️ File nginx $NGINX_CONF không tồn tại. Bỏ qua chỉnh port."
fi

### === 11. Cấu hình SSL nếu cần ===
if [ "$USE_SSL" = true ] && [ -n "$DOMAIN" ]; then
    echo "[9/9] 🔐 Setting up Let's Encrypt SSL..."
    sudo apt install -y certbot python3-certbot-nginx
    sudo certbot --nginx -d $DOMAIN
else
    echo "[9/9] 🔐 Skipping SSL. Dùng USE_SSL=true và nhập DOMAIN để bật."
fi

### ✅ DONE
echo -e "\n✅ ERPNext Production Ready!"
echo "🔗 Truy cập tại: http://<IP-address>:$SITE_PORT"
echo "🔧 Quản lý: sudo supervisorctl status"
echo "📁 Project path: /home/$ERP_USER/erpnext15"
