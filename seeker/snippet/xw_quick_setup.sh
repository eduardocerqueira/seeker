#date: 2025-06-17T16:45:13Z
#url: https://api.github.com/gists/d2f26c05f22dce60ed07ce3c68042df6
#owner: https://api.github.com/users/jkautto

#!/bin/bash
# Xwander AI v0.0.1 - Quick Setup Script
# Run this to set up Xwander AI in 2 minutes

set -e  # Exit on error

echo "🚀 Xwander AI v0.0.1 Quick Setup"
echo "================================"

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "❌ Please don't run as root"
   exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found"

# Create directory structure
echo "📁 Creating directory structure..."
sudo mkdir -p /srv/xwander/{data,logs}
sudo chown -R $USER:$USER /srv/xwander
cd /srv/xwander

# Create the main files
echo "📝 Creating xw.py..."
cat > xw.py << 'EOF'
#!/usr/bin/env python3
"""
Xwander AI v0.0.1 - Minimal Business Assistant
Detects booking emails and sends Slack notifications
"""

import json
import sys
import logging
from pathlib import Path
from datetime import datetime

__version__ = "0.0.1"

class XwCore:
    def __init__(self):
        self.setup_logging()
        self.config = self.load_config()
        logging.info(f"Xwander AI v{__version__} initialized")
        
    def setup_logging(self):
        """Basic logging setup"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/xw.log'),
                logging.StreamHandler()
            ]
        )
        
    def load_config(self):
        """Load configuration"""
        config_path = Path('config.json')
        if not config_path.exists():
            logging.error("config.json not found!")
            sys.exit(1)
            
        with open(config_path) as f:
            return json.load(f)
    
    def run(self):
        """Main execution"""
        try:
            # Check for new bookings
            from booking_monitor import BookingMonitor
            monitor = BookingMonitor(self.config)
            new_bookings = monitor.check()
            
            if new_bookings:
                logging.info(f"Found {len(new_bookings)} new booking(s)")
                self.notify_slack(new_bookings)
            else:
                logging.info("No new bookings found")
                
        except Exception as e:
            logging.error(f"Error in main execution: {e}")
            self.notify_error(str(e))
    
    def notify_slack(self, bookings):
        """Send booking notifications to Slack"""
        from slack_notify import SlackNotifier
        notifier = SlackNotifier(self.config['slack_webhook'])
        
        for booking in bookings:
            message = self.format_booking_message(booking)
            notifier.send(message)
            logging.info(f"Sent Slack notification for booking from {booking['from']}")
    
    def format_booking_message(self, booking):
        """Format booking for Slack"""
        return {
            "text": f"🎿 New Booking Inquiry",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*New Booking Inquiry*\n"
                                f"*From:* {booking['from']}\n"
                                f"*Subject:* {booking['subject']}\n"
                                f"*Date:* {booking['date']}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Preview:* {booking.get('preview', 'No preview available')}"
                    }
                }
            ]
        }
    
    def notify_error(self, error_msg):
        """Notify about errors via Slack"""
        from slack_notify import SlackNotifier
        notifier = SlackNotifier(self.config['slack_webhook'])
        
        message = {
            "text": f"❌ Xwander AI Error: {error_msg}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Error in Xwander AI*\n`{error_msg}`"
                    }
                }
            ]
        }
        notifier.send(message)

if __name__ == "__main__":
    xw = XwCore()
    xw.run()
EOF

echo "📝 Creating booking_monitor.py..."
cat > booking_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
Booking Monitor - Checks for new booking emails
"""

import imaplib
import email
from email.header import decode_header
import json
import logging
from pathlib import Path
from datetime import datetime

class BookingMonitor:
    def __init__(self, config):
        self.email_address = config['email']
        self.password = "**********"
        self.keywords = config.get('keywords', ['booking', 'reservation', 'book'])
        self.state_file = Path('data/booking_state.json')
        self.state = self.load_state()
        
    def load_state(self):
        """Load seen email IDs"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {"seen_ids": [], "last_check": None}
    
    def save_state(self):
        """Save seen email IDs"""
        self.state['last_check'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def check(self):
        """Check for new booking emails"""
        new_bookings = []
        
        try:
            # Connect to Gmail
            mail = imaplib.IMAP4_SSL('imap.gmail.com')
            mail.login(self.email_address, self.password)
            mail.select('inbox')
            
            # Search for emails with keywords
            for keyword in self.keywords:
                _, messages = mail.search(None, f'(BODY "{keyword}")')
                
                email_ids = messages[0].split()
                logging.info(f"Found {len(email_ids)} emails with keyword '{keyword}'")
                
                for email_id in email_ids:
                    # Skip if already seen
                    email_id_str = email_id.decode()
                    if email_id_str in self.state['seen_ids']:
                        continue
                    
                    # Fetch and parse email
                    booking_info = self._parse_email(mail, email_id)
                    if booking_info:
                        new_bookings.append(booking_info)
                        self.state['seen_ids'].append(email_id_str)
            
            mail.close()
            mail.logout()
            
            # Save state
            self.save_state()
            
        except Exception as e:
            logging.error(f"Error checking emails: {e}")
            raise
        
        return new_bookings
    
    def _parse_email(self, mail, email_id):
        """Parse email and extract booking info"""
        try:
            _, msg_data = mail.fetch(email_id, '(RFC822)')
            email_body = msg_data[0][1]
            message = email.message_from_bytes(email_body)
            
            # Extract basic info
            subject = self._decode_header(message['Subject'])
            from_addr = self._decode_header(message['From'])
            date = message['Date']
            
            # Get email preview (first 200 chars of body)
            body = self._get_email_body(message)
            preview = body[:200] + "..." if len(body) > 200 else body
            
            return {
                'id': email_id.decode(),
                'subject': subject,
                'from': from_addr,
                'date': date,
                'preview': preview,
                'body': body
            }
                
        except Exception as e:
            logging.error(f"Error parsing email {email_id}: {e}")
            
        return None
    
    def _decode_header(self, header):
        """Decode email header"""
        if header is None:
            return ""
            
        decoded = decode_header(header)
        result = []
        
        for text, charset in decoded:
            if isinstance(text, bytes):
                text = text.decode(charset or 'utf-8', errors='ignore')
            result.append(text)
            
        return ' '.join(result)
    
    def _get_email_body(self, message):
        """Extract email body text"""
        body = ""
        
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
            
        return body.strip()
EOF

echo "📝 Creating slack_notify.py..."
cat > slack_notify.py << 'EOF'
#!/usr/bin/env python3
"""
Slack Notifier - Sends messages to Slack via webhook
"""

import json
import urllib.request
import urllib.error
import logging

class SlackNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
        
    def send(self, message):
        """Send message to Slack"""
        try:
            # Handle both string and dict messages
            if isinstance(message, str):
                payload = {"text": message}
            else:
                payload = message
            
            # Prepare request
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            # Send request
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    logging.info("Successfully sent message to Slack")
                    return True
                    
        except Exception as e:
            logging.error(f"Error sending to Slack: {e}")
            return False
EOF

echo "📝 Creating config.json template..."
cat > config.json << 'EOF'
{
  "email": "bookings@xwander.fi",
  "password": "**********"
  "slack_webhook": "https://hooks.slack.com/services/YOUR/WEBHOOK/HERE",
  "keywords": [
    "booking",
    "reservation",
    "book",
    "tour",
    "safari",
    "northern lights",
    "husky",
    "snowshoe",
    "want to visit",
    "availability",
    "price",
    "cost"
  ]
}
EOF

# Make files executable
chmod +x xw.py booking_monitor.py slack_notify.py

echo ""
echo "✅ Files created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit config.json with your credentials:"
echo "   - Gmail app password (not your regular password)"
echo "   - Slack webhook URL"
echo ""
echo "2. Test manually:"
echo "   cd /srv/xwander && python3 xw.py"
echo ""
echo "3. Add to cron (runs every 10 minutes):"
echo "   crontab -e"
echo "   */10 * * * * cd /srv/xwander && /usr/bin/python3 xw.py >> logs/xw.log 2>&1"
echo ""
echo "4. Send a test email to bookings@xwander.fi with 'booking' in the subject"
echo ""
echo "📁 Installation location: /srv/xwander"
echo "📄 Logs will be in: /srv/xwander/logs/xw.log"
echo ""
echo "🎉 Setup complete! Don't forget to update config.json!"
EOF

chmod +x /tmp/xw_quick_setup.shp/xw_quick_setup.sh