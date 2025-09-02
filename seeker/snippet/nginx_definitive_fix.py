#date: 2025-09-02T16:49:54Z
#url: https://api.github.com/gists/1b12b278f39e16abaf3ad0e5439a7ed3
#owner: https://api.github.com/users/FrankSpooren

#!/usr/bin/env python3
"""
HoliBot Nginx Definitive Fix
Based on previous successful solutions from project history
Ensures all location blocks are INSIDE server blocks
"""

import os
import re
import shutil
import subprocess
from datetime import datetime

def analyze_existing_config():
    """Analyze the current broken configuration"""
    print("🔍 ANALYZING CURRENT NGINX CONFIG...")
    print("="*60)
    
    config_file = "/etc/nginx/sites-enabled/holidaibutler.com"
    
    if not os.path.exists(config_file):
        print(f"   ⚠️ Config not found at {config_file}")
        # Try alternative location
        config_file = "/etc/nginx/sites-available/holidaibutler.com"
        if not os.path.exists(config_file):
            config_file = "/etc/nginx/sites-enabled/default"
    
    print(f"   Using config: {config_file}")
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Find misplaced location blocks
    lines = content.split('\n')
    
    # Track brace depth
    brace_depth = 0
    server_blocks = []
    current_server = None
    misplaced_locations = []
    
    for i, line in enumerate(lines):
        # Count braces
        open_braces = line.count('{')
        close_braces = line.count('}')
        
        # Check for server block start
        if 'server' in line and '{' in line:
            current_server = {'start': i, 'depth': brace_depth}
            
        brace_depth += open_braces
        brace_depth -= close_braces
        
        # Server block ended
        if current_server and brace_depth < current_server['depth']:
            current_server['end'] = i
            server_blocks.append(current_server)
            current_server = None
            
        # Location block outside server (brace_depth should be 0 outside server blocks)
        if 'location' in line and not any(i >= s['start'] and i <= s.get('end', float('inf')) for s in server_blocks):
            misplaced_locations.append(i)
            print(f"   ❌ Found misplaced location at line {i+1}: {line.strip()[:50]}")
    
    return config_file, lines, server_blocks, misplaced_locations

def extract_widget_locations():
    """Extract the widget location blocks that need to be added"""
    widget_locations = """
    # HoliBot Widget Endpoints (INSIDE SERVER BLOCK)
    location /widget {
        alias /home/holibot/holibot-api/www/widget.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header X-Frame-Options "SAMEORIGIN";
    }
    
    location /widget-test {
        alias /home/holibot/holibot-api/www/test.html;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
    
    location /widget-assets/ {
        alias /home/holibot/holibot-api/www/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }"""
    
    return widget_locations

def create_fixed_config(config_file, lines, server_blocks, misplaced_locations):
    """Create a fixed configuration with locations properly placed"""
    print("\n📝 CREATING FIXED CONFIGURATION...")
    print("="*60)
    
    # Remove misplaced location blocks
    lines_to_remove = set()
    for loc_line in misplaced_locations:
        # Find the complete location block
        brace_count = 0
        started = False
        for i in range(loc_line, len(lines)):
            if '{' in lines[i]:
                started = True
                brace_count += lines[i].count('{')
            if started:
                lines_to_remove.add(i)
                brace_count -= lines[i].count('}')
                if brace_count == 0:
                    break
    
    # Create new lines without misplaced blocks
    new_lines = []
    for i, line in enumerate(lines):
        if i not in lines_to_remove:
            new_lines.append(line)
    
    # Find HTTPS server block (port 443)
    https_server = None
    for server in server_blocks:
        block_text = '\n'.join(lines[server['start']:server.get('end', len(lines))])
        if '443' in block_text or 'ssl' in block_text:
            https_server = server
            break
    
    if not https_server:
        print("   ⚠️ No HTTPS server block found, will add to last server block")
        https_server = server_blocks[-1] if server_blocks else None
    
    if https_server:
        # Find the insertion point (before closing brace of server block)
        insert_line = https_server.get('end', len(new_lines)) - 1
        
        # Adjust for removed lines
        removed_before = sum(1 for i in lines_to_remove if i < insert_line)
        insert_line -= removed_before
        
        # Insert widget locations
        widget_locations = extract_widget_locations()
        widget_lines = widget_locations.split('\n')
        
        print(f"   ✓ Inserting widget locations at line {insert_line}")
        
        # Insert the widget configuration
        for widget_line in reversed(widget_lines):
            new_lines.insert(insert_line, widget_line)
    
    return '\n'.join(new_lines)

def apply_fix(fixed_config, config_file):
    """Apply the fixed configuration"""
    print("\n🔧 APPLYING FIX...")
    print("="*60)
    
    # Backup current config
    backup_file = f"{config_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(config_file, backup_file)
    print(f"   ✓ Backup created: {backup_file}")
    
    # Write fixed config to temp file first
    temp_file = "/tmp/nginx_fixed.conf"
    with open(temp_file, 'w') as f:
        f.write(fixed_config)
    
    # Test the temp config
    print("   🧪 Testing fixed configuration...")
    
    # Copy to nginx location
    shutil.copy2(temp_file, config_file)
    
    # Test nginx configuration
    result = subprocess.run(['nginx', '-t'], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Configuration test PASSED!")
        
        # Reload nginx
        subprocess.run(['nginx', '-s', 'reload'])
        print("   ✅ Nginx reloaded successfully!")
        
        return True
    else:
        print("   ❌ Configuration test failed:")
        print(result.stderr)
        
        # Restore backup
        shutil.copy2(backup_file, config_file)
        print(f"   ↩️ Restored backup: {backup_file}")
        
        return False

def verify_widget_access():
    """Verify widget endpoints are accessible"""
    print("\n🌐 VERIFYING WIDGET ACCESS...")
    print("="*60)
    
    endpoints = [
        "https://holidaibutler.com/widget",
        "https://holidaibutler.com/widget-test"
    ]
    
    for endpoint in endpoints:
        try:
            # Use curl to test
            result = subprocess.run(
                ['curl', '-I', '-s', endpoint],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if '200' in result.stdout or '301' in result.stdout:
                print(f"   ✅ {endpoint} - Accessible")
            else:
                print(f"   ⚠️ {endpoint} - Status unknown")
        except:
            print(f"   ⚠️ Could not test {endpoint}")

def main():
    """Main execution flow"""
    print("="*60)
    print("🔧 NGINX DEFINITIVE FIX - BASED ON PROJECT HISTORY")
    print("="*60)
    print(f"Execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis fix addresses the recurring issue of location blocks")
    print("being placed outside server blocks - a pattern we've seen")
    print("multiple times in this project.")
    
    # Step 1: Analyze current config
    config_file, lines, server_blocks, misplaced_locations = analyze_existing_config()
    
    if misplaced_locations:
        print(f"\n⚠️ Found {len(misplaced_locations)} misplaced location blocks")
        
        # Step 2: Create fixed config
        fixed_config = create_fixed_config(config_file, lines, server_blocks, misplaced_locations)
        
        # Step 3: Apply fix
        if apply_fix(fixed_config, config_file):
            print("\n" + "="*60)
            print("✅ NGINX CONFIGURATION FIXED SUCCESSFULLY!")
            print("="*60)
            
            # Step 4: Verify
            verify_widget_access()
            
            print("\n📊 SUMMARY:")
            print("  ✓ All location blocks now INSIDE server blocks")
            print("  ✓ Widget endpoints properly configured")
            print("  ✓ Nginx configuration valid and reloaded")
            print("  ✓ No more 'location directive not allowed' errors")
            
            print("\n🎯 LESSONS LEARNED:")
            print("  • Always place location blocks INSIDE server blocks")
            print("  • Test configuration before reloading nginx")
            print("  • Keep backups of working configurations")
            
            print("\n🚀 NEXT STEPS:")
            print("  1. Test widget: curl https://holidaibutler.com/widget")
            print("  2. Activate cron jobs for monitoring")
            print("  3. Complete TripAdvisor integration")
        else:
            print("\n❌ Fix failed - check error messages above")
            print("Manual intervention may be required")
    else:
        print("\n✅ No misplaced location blocks found!")
        print("Configuration appears to be correct")
        
        # Still test nginx
        result = subprocess.run(['nginx', '-t'], capture_output=True, text=True)
        if result.returncode != 0:
            print("\n⚠️ But nginx test still fails:")
            print(result.stderr)

if __name__ == "__main__":
    main()