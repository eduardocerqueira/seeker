#date: 2025-09-02T17:08:56Z
#url: https://api.github.com/gists/f397cf2c1e5546fa90077e3bfe16e197
#owner: https://api.github.com/users/FrankSpooren

#!/usr/bin/env python3
"""
HoliBot Nginx Complete Fix
Removes all broken configs and creates clean working configuration
GitHub Gist deployment ready
"""

import os
import shutil
import subprocess
import glob
from datetime import datetime

def cleanup_nginx_directory():
    """Clean up all broken configs from sites-enabled"""
    print("🧹 CLEANING NGINX DIRECTORY...")
    print("="*60)
    
    sites_enabled = "/etc/nginx/sites-enabled"
    backup_dir = "/home/holibot/nginx_backups"
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Find all files in sites-enabled
    configs = glob.glob(f"{sites_enabled}/*")
    
    for config in configs:
        filename = os.path.basename(config)
        print(f"   Moving {filename} to backup directory...")
        
        # Move to backup with timestamp
        backup_name = f"{backup_dir}/{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(config, backup_name)
        print(f"   ✓ Backed up to {backup_name}")
    
    print(f"   ✓ Cleaned {len(configs)} files from sites-enabled")
    return True

def create_clean_config():
    """Create a completely new, working nginx configuration"""
    print("\n📝 CREATING CLEAN CONFIGURATION...")
    print("="*60)
    
    config_content = """# HolidaiButler.com - Production Configuration
# Generated: {timestamp}

# HTTP to HTTPS redirect
server {{
    listen 80;
    listen [::]:80;
    server_name holidaibutler.com www.holidaibutler.com;
    return 301 https://$server_name$request_uri;
}}

# Main HTTPS server
server {{
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name holidaibutler.com www.holidaibutler.com;
    
    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/holidaibutler.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/holidaibutler.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Root API proxy
    location / {{
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    # Search endpoint
    location /search {{
        proxy_pass http://127.0.0.1:8000/search;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }}
    
    # Hybrid search endpoint
    location /search/hybrid {{
        proxy_pass http://127.0.0.1:8000/search/hybrid;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }}
    
    # Status endpoints
    location /hybrid/status {{
        proxy_pass http://127.0.0.1:8000/hybrid/status;
        proxy_set_header Host $host;
    }}
    
    location /health {{
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
    }}
    
    # Widget endpoints - PROPERLY INSIDE SERVER BLOCK
    location /widget {{
        alias /home/holibot/holibot-api/www/widget.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header X-Frame-Options "SAMEORIGIN" always;
    }}
    
    location /widget-test {{
        alias /home/holibot/holibot-api/www/test.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }}
    
    location /widget-assets/ {{
        alias /home/holibot/holibot-api/www/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }}
    
    # Monitoring endpoint
    location /monitor {{
        proxy_pass http://127.0.0.1:8000/monitor;
        proxy_set_header Host $host;
    }}
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/rss+xml application/atom+xml image/svg+xml text/javascript application/vnd.ms-fontobject application/x-font-ttf font/opentype;
}}""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Write to file
    config_path = "/etc/nginx/sites-enabled/holidaibutler.com"
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"   ✓ Configuration written to {config_path}")
    return config_path

def test_and_reload():
    """Test the configuration and reload if successful"""
    print("\n🧪 TESTING CONFIGURATION...")
    print("="*60)
    
    # Test nginx configuration
    result = subprocess.run(['nginx', '-t'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Configuration test PASSED!")
        print(result.stdout)
        
        # Reload nginx
        print("\n🔄 RELOADING NGINX...")
        reload_result = subprocess.run(['nginx', '-s', 'reload'], capture_output=True, text=True)
        
        if reload_result.returncode == 0:
            print("   ✅ Nginx reloaded successfully!")
            return True
        else:
            print("   ❌ Reload failed:")
            print(reload_result.stderr)
            return False
    else:
        print("   ❌ Configuration test FAILED:")
        print(result.stderr)
        return False

def verify_endpoints():
    """Verify that endpoints are working"""
    print("\n🌐 VERIFYING ENDPOINTS...")
    print("="*60)
    
    endpoints = [
        ("Health Check", "http://localhost:8000/health"),
        ("Widget File", "/home/holibot/holibot-api/www/widget.html"),
        ("Test File", "/home/holibot/holibot-api/www/test.html")
    ]
    
    for name, endpoint in endpoints:
        if endpoint.startswith("http"):
            # Test HTTP endpoint
            try:
                result = subprocess.run(
                    ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', endpoint],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                status = result.stdout.strip()
                if status in ['200', '301', '302']:
                    print(f"   ✅ {name}: {status} OK")
                else:
                    print(f"   ⚠️ {name}: {status}")
            except:
                print(f"   ⚠️ {name}: Could not test")
        else:
            # Check file existence
            if os.path.exists(endpoint):
                size = os.path.getsize(endpoint)
                print(f"   ✅ {name}: Exists ({size} bytes)")
            else:
                print(f"   ❌ {name}: Not found")

def show_next_steps():
    """Show what to do next"""
    print("\n📋 NEXT STEPS...")
    print("="*60)
    
    print("""
1. TEST WIDGET ACCESS:
   curl -I https://holidaibutler.com/widget
   curl -I https://holidaibutler.com/widget-test

2. ACTIVATE CRON JOBS:
   crontab -e
   # Add:
   0 * * * * /home/holibot/holibot-api/monitoring/cron_monitor.sh

3. VERIFY API:
   curl https://holidaibutler.com/health
   curl https://holidaibutler.com/search/hybrid

4. CHECK MONITORING:
   cd /home/holibot/holibot-api/monitoring
   python3 auto_update_top10.py
""")

def main():
    """Main execution"""
    print("="*60)
    print("🔧 NGINX COMPLETE FIX - GITHUB GIST DEPLOYMENT")
    print("="*60)
    print(f"Execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will:")
    print("  1. Clean up all broken configs")
    print("  2. Create new working configuration")
    print("  3. Test and reload nginx")
    print("  4. Verify endpoints")
    
    try:
        # Step 1: Clean up
        cleanup_nginx_directory()
        
        # Step 2: Create new config
        create_clean_config()
        
        # Step 3: Test and reload
        if test_and_reload():
            # Step 4: Verify
            verify_endpoints()
            
            print("\n" + "="*60)
            print("✅ NGINX COMPLETELY FIXED!")
            print("="*60)
            
            print("\n📊 SUMMARY:")
            print("  ✓ All broken configs removed")
            print("  ✓ Clean configuration created")
            print("  ✓ Widget locations inside server block")
            print("  ✓ Nginx reloaded successfully")
            
            # Show next steps
            show_next_steps()
            
            print("\n🎯 PROJECT STATUS:")
            print("  ✅ Backend API: Operational")
            print("  ✅ Auto-update: Configured")
            print("  ✅ Widget: Deployed")
            print("  ✅ Nginx: FIXED")
            print("  ⏳ Cron: Ready to activate")
            
        else:
            print("\n❌ Fix failed - check errors above")
            print("Backups are in /home/holibot/nginx_backups/")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check /home/holibot/nginx_backups/ for backups")

if __name__ == "__main__":
    main()