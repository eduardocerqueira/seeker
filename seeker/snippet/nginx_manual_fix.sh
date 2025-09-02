#date: 2025-09-02T16:54:39Z
#url: https://api.github.com/gists/8edb297be5c1d4c9a56977b94344b4c6
#owner: https://api.github.com/users/FrankSpooren

#!/bin/bash
# Nginx Manual Fix - Complete Clean Configuration
# This creates a working nginx config from scratch

echo "🔧 NGINX MANUAL FIX - CREATING CLEAN CONFIG"
echo "=========================================="

# Step 1: Backup current broken config
echo "📁 Creating backup..."
cp /etc/nginx/sites-enabled/holidaibutler.com /home/holibot/nginx_broken_$(date +%Y%m%d_%H%M%S).backup
echo "   ✓ Backup saved to /home/holibot/"

# Step 2: Show what's wrong
echo ""
echo "📋 Current broken config (first 20 lines):"
head -20 /etc/nginx/sites-enabled/holidaibutler.com

# Step 3: Create completely new, working config
echo ""
echo "📝 Creating new clean configuration..."

cat > /etc/nginx/sites-enabled/holidaibutler.com << 'EOF'
# HolidaiButler.com - Clean Nginx Configuration
# Created: September 2025

# HTTP to HTTPS redirect
server {
    listen 80;
    listen [::]:80;
    server_name holidaibutler.com www.holidaibutler.com;
    
    # Redirect all HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

# HTTPS Server Block
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name holidaibutler.com www.holidaibutler.com;
    
    # SSL Certificate paths
    ssl_certificate /etc/letsencrypt/live/holidaibutler.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/holidaibutler.com/privkey.pem;
    
    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    
    # Security Headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Main HoliBot API proxy
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API endpoints
    location /search {
        proxy_pass http://127.0.0.1:8000/search;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /search/hybrid {
        proxy_pass http://127.0.0.1:8000/search/hybrid;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /hybrid/status {
        proxy_pass http://127.0.0.1:8000/hybrid/status;
        proxy_set_header Host $host;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
    }
    
    # HoliBot Widget - CORRECTLY INSIDE SERVER BLOCK
    location /widget {
        alias /home/holibot/holibot-api/www/widget.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header X-Frame-Options "SAMEORIGIN" always;
    }
    
    location /widget-test {
        alias /home/holibot/holibot-api/www/test.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
    
    location /widget-assets/ {
        alias /home/holibot/holibot-api/www/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    # Monitoring endpoints (if needed)
    location /monitor {
        proxy_pass http://127.0.0.1:8000/monitor;
        proxy_set_header Host $host;
    }
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/rss+xml application/atom+xml image/svg+xml text/javascript application/vnd.ms-fontobject application/x-font-ttf font/opentype;
}
EOF

echo "   ✓ New configuration created"

# Step 4: Test the new configuration
echo ""
echo "🧪 Testing new configuration..."
nginx -t

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Configuration test PASSED!"
    echo ""
    echo "🔄 Reloading nginx..."
    nginx -s reload
    
    if [ $? -eq 0 ]; then
        echo "   ✓ Nginx reloaded successfully!"
        echo ""
        echo "=========================================="
        echo "✅ NGINX FIXED SUCCESSFULLY!"
        echo "=========================================="
        echo ""
        echo "📊 What was fixed:"
        echo "   • Location blocks now properly inside server blocks"
        echo "   • Widget endpoints correctly configured"
        echo "   • SSL configuration in place"
        echo "   • All API endpoints proxied correctly"
        echo ""
        echo "🌐 Test your endpoints:"
        echo "   curl https://holidaibutler.com/health"
        echo "   curl https://holidaibutler.com/widget"
        echo "   curl https://holidaibutler.com/widget-test"
    else
        echo "   ❌ Reload failed!"
    fi
else
    echo ""
    echo "❌ Configuration test FAILED!"
    echo "Check the error messages above"
    echo ""
    echo "📝 To restore backup:"
    echo "   cp /home/holibot/nginx_broken_*.backup /etc/nginx/sites-enabled/holidaibutler.com"
fi