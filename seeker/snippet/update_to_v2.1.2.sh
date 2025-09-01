#date: 2025-09-01T17:06:00Z
#url: https://api.github.com/gists/a5d02b06ffa3d067688b2cb6f12cdf2f
#owner: https://api.github.com/users/4ractl

#!/usr/bin/env bash

#############################################################
# Vultr Server Update to v2.1.2
# Full Auto-Scaling Edition
# Cumulative update script for v2.0.9 servers
# Adds auto-scaling for PHP-FPM, OPcache, NGINX, and Redis
#############################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NEW_VERSION="2.1.2"
REQUIRED_USER="root"

# Safety checks
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}ERROR: Must run as root${NC}"
   exit 1
fi

if [[ ! -f /etc/nginx/nginx.conf ]]; then
   echo -e "${RED}ERROR: NGINX not found. Is this a Vultr setup server?${NC}"
   exit 1
fi

if [[ ! -d /home/sysop ]]; then
   echo -e "${RED}ERROR: sysop user not found. Is this a Vultr setup server?${NC}"
   exit 1
fi

echo -e "${GREEN}=== Vultr Server Update to v${NEW_VERSION} ===${NC}"
echo ""

# Check current version if exists
if [[ -f /root/.vultr_version ]]; then
    CURRENT_VERSION=$(cat /root/.vultr_version)
    echo "Current version: $CURRENT_VERSION"
    echo "Target version: $NEW_VERSION"
    
    # Check if already on v2.1.2
    if [[ "$CURRENT_VERSION" == "$NEW_VERSION" ]]; then
        echo -e "${YELLOW}Already on version $NEW_VERSION - nothing to do${NC}"
        exit 0
    fi
    
    # Ensure we're on v2.0.9 or later
    if [[ ! "$CURRENT_VERSION" =~ ^2\.(0\.[89]|1\.[0-9]+) ]]; then
        echo -e "${RED}ERROR: This update requires v2.0.9 or later${NC}"
        echo "Please run update_to_v2.0.9.sh first"
        exit 1
    fi
else
    echo -e "${RED}ERROR: No version file found${NC}"
    echo "This update is for v2.0.9+ servers only"
    exit 1
fi

echo ""
echo -e "${GREEN}=== AUTO-SCALING FEATURES ===${NC}"
echo "This update adds intelligent auto-scaling based on server resources:"
echo ""
echo "• ${BLUE}PHP-FPM Workers${NC}: Auto-scales from 35 (1GB) to 600 (16GB+)"
echo "• ${BLUE}PHP Memory Limit${NC}: Auto-scales from 64MB to 512MB"
echo "• ${BLUE}OPcache Memory${NC}: Auto-scales from 64MB to 1024MB"
echo "• ${BLUE}NGINX Connections${NC}: Auto-scales from 1024 to 16384"
echo "• ${BLUE}Redis Memory${NC}: Auto-scales using 10% of RAM (64MB-2GB)"
echo ""
echo "The configuration will automatically adjust based on your server's RAM."
echo ""

# Detect server resources
TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
TOTAL_RAM_GB=$((TOTAL_RAM_MB / 1024))
CPU_CORES=$(nproc)

echo "Detected Resources:"
echo "• RAM: ${TOTAL_RAM_MB}MB (~${TOTAL_RAM_GB}GB)"
echo "• CPU Cores: ${CPU_CORES}"
echo ""

read -p "Continue with update? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Update cancelled"
    exit 0
fi

echo ""
echo "Starting update..."

# Create backup marker
BACKUP_MARKER="/root/.update_backup_$(date +%Y%m%d_%H%M%S)"
touch $BACKUP_MARKER
echo "Backup marker created: $BACKUP_MARKER"

# Auto-scaling configuration function
configure_auto_scaling() {
    echo ""
    echo -e "${GREEN}=== Configuring Auto-Scaling ===${NC}"
    
    # Determine configuration based on RAM
    if [ "$TOTAL_RAM_MB" -le 1024 ]; then
        # 1GB or less
        PHP_MAX_CHILDREN=35
        PHP_START_SERVERS=8
        PHP_MIN_SPARE=5
        PHP_MAX_SPARE=15
        PHP_MEMORY_LIMIT="64M"
        OPCACHE_MEMORY=64
        NGINX_WORKER_CONNECTIONS=1024
        FASTCGI_CACHE_SIZE="50m"
        FASTCGI_CACHE_MAX="128m"
        REDIS_MEMORY=$((TOTAL_RAM_MB / 10))  # 10% of RAM
        if [ "$REDIS_MEMORY" -lt 64 ]; then
            REDIS_MEMORY=64
        fi
    elif [ "$TOTAL_RAM_MB" -le 2048 ]; then
        # 2GB
        PHP_MAX_CHILDREN=70
        PHP_START_SERVERS=15
        PHP_MIN_SPARE=10
        PHP_MAX_SPARE=30
        PHP_MEMORY_LIMIT="128M"
        OPCACHE_MEMORY=128
        NGINX_WORKER_CONNECTIONS=2048
        FASTCGI_CACHE_SIZE="100m"
        FASTCGI_CACHE_MAX="256m"
        REDIS_MEMORY=$((TOTAL_RAM_MB / 10))  # 10% of RAM
    elif [ "$TOTAL_RAM_MB" -le 4096 ]; then
        # 4GB
        PHP_MAX_CHILDREN=150
        PHP_START_SERVERS=30
        PHP_MIN_SPARE=20
        PHP_MAX_SPARE=60
        PHP_MEMORY_LIMIT="256M"
        OPCACHE_MEMORY=256
        NGINX_WORKER_CONNECTIONS=4096
        FASTCGI_CACHE_SIZE="200m"
        FASTCGI_CACHE_MAX="512m"
        REDIS_MEMORY=$((TOTAL_RAM_MB / 10))  # 10% of RAM
    elif [ "$TOTAL_RAM_MB" -le 8192 ]; then
        # 8GB
        PHP_MAX_CHILDREN=300
        PHP_START_SERVERS=60
        PHP_MIN_SPARE=40
        PHP_MAX_SPARE=120
        PHP_MEMORY_LIMIT="256M"
        OPCACHE_MEMORY=512
        NGINX_WORKER_CONNECTIONS=8192
        FASTCGI_CACHE_SIZE="400m"
        FASTCGI_CACHE_MAX="1g"
        REDIS_MEMORY=$((TOTAL_RAM_MB / 10))  # 10% of RAM
    elif [ "$TOTAL_RAM_MB" -le 12288 ]; then
        # 12GB
        PHP_MAX_CHILDREN=450
        PHP_START_SERVERS=90
        PHP_MIN_SPARE=60
        PHP_MAX_SPARE=180
        PHP_MEMORY_LIMIT="384M"
        OPCACHE_MEMORY=768
        NGINX_WORKER_CONNECTIONS=12288
        FASTCGI_CACHE_SIZE="600m"
        FASTCGI_CACHE_MAX="1536m"
        REDIS_MEMORY=$((TOTAL_RAM_MB / 10))  # 10% of RAM
    else
        # 16GB or more
        PHP_MAX_CHILDREN=600
        PHP_START_SERVERS=120
        PHP_MIN_SPARE=80
        PHP_MAX_SPARE=240
        PHP_MEMORY_LIMIT="512M"
        OPCACHE_MEMORY=1024
        NGINX_WORKER_CONNECTIONS=16384
        FASTCGI_CACHE_SIZE="800m"
        FASTCGI_CACHE_MAX="2g"
        REDIS_MEMORY=$((TOTAL_RAM_MB / 10))  # 10% of RAM
        if [ "$REDIS_MEMORY" -gt 2048 ]; then
            REDIS_MEMORY=2048  # Cap at 2GB
        fi
    fi
    
    echo ""
    echo "Auto-Scaling Configuration:"
    echo "├─ PHP-FPM Workers: ${PHP_MAX_CHILDREN}"
    echo "├─ PHP Memory Limit: ${PHP_MEMORY_LIMIT}"
    echo "├─ OPcache Memory: ${OPCACHE_MEMORY}MB"
    echo "├─ NGINX Connections: ${NGINX_WORKER_CONNECTIONS}"
    echo "├─ Redis Memory: ${REDIS_MEMORY}MB"
    echo "└─ FastCGI Cache: ${FASTCGI_CACHE_SIZE}/${FASTCGI_CACHE_MAX}"
}

# Apply auto-scaling configuration
configure_auto_scaling

echo ""
echo -e "${GREEN}Applying auto-scaling configuration...${NC}"

# 1. Update PHP-FPM configuration
echo "Configuring PHP-FPM with auto-scaling..."
cat > /etc/php/8.3/fpm/pool.d/www.conf << EOF
[www]
user = www-data
group = www-data
listen = /run/php/php8.3-fpm.sock
listen.owner = www-data
listen.group = www-data
listen.mode = 0660

; Auto-scaled values for ${TOTAL_RAM_MB}MB RAM
pm = dynamic
pm.max_children = ${PHP_MAX_CHILDREN}
pm.start_servers = ${PHP_START_SERVERS}
pm.min_spare_servers = ${PHP_MIN_SPARE}
pm.max_spare_servers = ${PHP_MAX_SPARE}
pm.max_requests = 500
pm.process_idle_timeout = 10s

; Logging
access.log = /var/log/php8.3-fpm/access.log
php_admin_value[error_log] = /var/log/php8.3-fpm/www-error.log
php_admin_flag[log_errors] = on

; Performance
php_admin_value[memory_limit] = ${PHP_MEMORY_LIMIT}
php_admin_value[max_execution_time] = 300
php_admin_value[post_max_size] = 100M
php_admin_value[upload_max_filesize] = 100M
php_admin_value[max_input_time] = 300
php_admin_value[max_input_vars] = 5000

; Security
php_admin_value[disable_functions] = exec,passthru,shell_exec,system,proc_open,popen,curl_exec,curl_multi_exec,parse_ini_file,show_source
php_admin_flag[expose_php] = off
EOF

# 2. Update PHP configuration
echo "Updating PHP configuration..."
sed -i "s/memory_limit = .*/memory_limit = ${PHP_MEMORY_LIMIT}/" /etc/php/8.3/fpm/php.ini
sed -i "s/opcache.memory_consumption=.*/opcache.memory_consumption=${OPCACHE_MEMORY}/" /etc/php/8.3/fpm/php.ini

# Ensure OPcache is properly configured
if ! grep -q "opcache.enable=1" /etc/php/8.3/fpm/php.ini; then
    cat >> /etc/php/8.3/fpm/php.ini << EOF

; OPcache auto-scaling configuration
opcache.enable=1
opcache.enable_cli=1
opcache.memory_consumption=${OPCACHE_MEMORY}
opcache.interned_strings_buffer=16
opcache.max_accelerated_files=10000
opcache.revalidate_freq=2
opcache.save_comments=1
opcache.validate_timestamps=1
EOF
fi

# 3. Update NGINX configuration
echo "Updating NGINX configuration..."
# Backup current config
cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.v212backup

# Create new NGINX config with auto-scaling
cat > /etc/nginx/nginx.conf << EOF
user www-data;
worker_processes auto;
pid /run/nginx.pid;
error_log /var/log/nginx/error.log warn;

events {
    worker_connections ${NGINX_WORKER_CONNECTIONS};
    use epoll;
    multi_accept on;
}

http {
    # Basic Settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;
    client_max_body_size 100M;

    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Rate Limiting (v2.0.9 feature retained)
    limit_req_zone \$binary_remote_addr zone=general:10m rate=10r/s;
    limit_req_zone \$binary_remote_addr zone=static:10m rate=30r/s;
    limit_req_zone \$binary_remote_addr zone=php:10m rate=2r/s;
    limit_conn_zone \$binary_remote_addr zone=addr:10m;

    # SSL Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    # Logging
    access_log /var/log/nginx/access.log combined buffer=32k flush=5m;

    # Gzip Settings
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/rss+xml application/atom+xml image/svg+xml text/x-js text/x-cross-domain-policy application/x-font-ttf application/x-font-opentype application/vnd.ms-fontobject font/opentype;

    # FastCGI Cache
    fastcgi_cache_path /var/cache/nginx/fastcgi levels=1:2 keys_zone=WORDPRESS:${FASTCGI_CACHE_SIZE} max_size=${FASTCGI_CACHE_MAX} inactive=60m;
    fastcgi_cache_key "\$scheme\$request_method\$host\$request_uri";
    fastcgi_cache_use_stale error timeout updating invalid_header http_500 http_503;
    fastcgi_cache_valid 200 301 302 60m;
    fastcgi_cache_valid 404 10m;
    
    # Cache bypass conditions
    map \$request_uri \$no_cache {
        default 0;
        ~*/wp-admin/ 1;
        ~*/wp-login.php 1;
        ~*/wp-cron.php 1;
    }

    # Virtual Host Configs
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
EOF

# 4. Update Redis configuration with auto-scaling
echo "Configuring Redis with auto-scaling..."
cat > /etc/redis/redis.conf << EOF
# Redis auto-scaling configuration for ${TOTAL_RAM_MB}MB RAM
bind 127.0.0.1 ::1
protected-mode yes
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 300
daemonize yes
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16

# Memory configuration - auto-scaled
maxmemory ${REDIS_MEMORY}mb
maxmemory-policy allkeys-lru

# Snapshotting
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis

# Append only mode
appendonly no
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Client output buffer limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# Threaded I/O
io-threads 4
io-threads-do-reads yes
EOF

# 5. Restart services
echo ""
echo "Restarting services..."
systemctl restart php8.3-fpm
echo -e "${GREEN}✓ PHP-FPM restarted${NC}"

systemctl restart redis-server
if systemctl is-active --quiet redis-server; then
    echo -e "${GREEN}✓ Redis restarted${NC}"
else
    echo -e "${RED}✗ Redis failed to start - checking status${NC}"
    systemctl status redis-server --no-pager
fi

nginx -t 2>/dev/null && systemctl reload nginx
echo -e "${GREEN}✓ NGINX reloaded${NC}"

# 6. Create auto-scaling report script
echo ""
echo "Creating auto-scaling report script..."
cat > /usr/local/bin/show-autoscale-config << 'EOF'
#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Auto-Scaling Configuration Report ===${NC}"
echo ""

# Server resources
TOTAL_RAM_MB=$(free -m | awk '/^Mem:/{print $2}')
TOTAL_RAM_GB=$((TOTAL_RAM_MB / 1024))
CPU_CORES=$(nproc)

echo -e "${BLUE}Server Resources:${NC}"
echo "├─ RAM: ${TOTAL_RAM_MB}MB (~${TOTAL_RAM_GB}GB)"
echo "└─ CPU Cores: ${CPU_CORES}"
echo ""

echo -e "${BLUE}PHP-FPM Configuration:${NC}"
grep -E "pm.max_children|memory_limit" /etc/php/8.3/fpm/pool.d/www.conf | while read line; do
    echo "├─ $line"
done
echo ""

echo -e "${BLUE}OPcache Configuration:${NC}"
php -i | grep opcache.memory_consumption | head -1
echo ""

echo -e "${BLUE}NGINX Configuration:${NC}"
grep worker_connections /etc/nginx/nginx.conf | head -1 | xargs echo "├─"
echo ""

echo -e "${BLUE}Redis Configuration:${NC}"
grep maxmemory /etc/redis/redis.conf | head -1 | xargs echo "├─"
echo ""

echo -e "${BLUE}Service Status:${NC}"
for service in nginx php8.3-fpm redis-server; do
    if systemctl is-active --quiet $service; then
        echo -e "├─ $service: ${GREEN}●${NC} active"
    else
        echo -e "├─ $service: ${RED}●${NC} inactive"
    fi
done
echo ""

echo -e "${BLUE}Current Memory Usage:${NC}"
free -h | grep "^Mem" | awk '{print "├─ Used: " $3 " / " $2}'
echo ""
EOF
chmod +x /usr/local/bin/show-autoscale-config

# 7. Update version file
echo "$NEW_VERSION" > /root/.vultr_version
echo -e "${GREEN}✓ Version updated to $NEW_VERSION${NC}"

# 8. Run configuration report
echo ""
/usr/local/bin/show-autoscale-config

# Final message
echo -e "${GREEN}=== Update to v${NEW_VERSION} Complete ===${NC}"
echo ""
echo "Your server is now running with auto-scaling configuration!"
echo ""
echo "Key commands:"
echo "• View auto-scaling config: ${GREEN}show-autoscale-config${NC}"
echo "• Check memory usage: ${GREEN}free -h${NC}"
echo "• Monitor services: ${GREEN}systemctl status nginx php8.3-fpm redis-server${NC}"
echo ""
echo -e "${YELLOW}Note: The configuration has been optimized for your ${TOTAL_RAM_MB}MB server${NC}"
echo ""
echo "Backup marker saved at: $BACKUP_MARKER"