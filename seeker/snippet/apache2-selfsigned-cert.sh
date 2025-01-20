#date: 2025-01-20T16:56:30Z
#url: https://api.github.com/gists/f45486ac95f022f772bf796b583c4e17
#owner: https://api.github.com/users/hardenedpenguin

#!/bin/sh

# Variables
CERT_DIR="/etc/ssl/certs"
KEY_DIR="/etc/ssl/private"
DOMAIN="example.com"
CERT_FILE="$CERT_DIR/${DOMAIN}.crt"
KEY_FILE="$KEY_DIR/${DOMAIN}.key"
DAYS_VALID=365

# Ensure the script is run as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    exit 1
fi

# Check if OpenSSL is installed
if ! command -v openssl >/dev/null 2>&1; then
    echo "OpenSSL is required but not installed. Installing..."
    apt update && apt install -y openssl || {
        echo "Failed to install OpenSSL. Exiting."
        exit 1
    }
fi

# Generate private key
echo "Generating private key..."
openssl genrsa -out "$KEY_FILE" 2048 || {
    echo "Failed to generate private key. Exiting."
    exit 1
}

# Set appropriate permissions for the private key
chmod 600 "$KEY_FILE"

# Generate self-signed certificate
echo "Generating self-signed certificate..."
openssl req -new -x509 -days "$DAYS_VALID" -key "$KEY_FILE" -out "$CERT_FILE" \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=$DOMAIN" || {
    echo "Failed to generate certificate. Exiting."
    exit 1
}

# Set permissions for the certificate
chmod 644 "$CERT_FILE"

# Verify Apache2 is installed
if ! command -v apache2 >/dev/null 2>&1; then
    echo "Apache2 is not installed. Please install it before proceeding."
    exit 1
fi

# Enable SSL module in Apache2
echo "Enabling SSL module in Apache2..."
a2enmod ssl || {
    echo "Failed to enable SSL module. Exiting."
    exit 1
}

# Create a default SSL configuration for Apache2 if it doesn't exist
SSL_CONF="/etc/apache2/sites-available/default-ssl.conf"
if [ ! -f "$SSL_CONF" ]; then
    echo "Creating default SSL configuration for Apache2..."
    cat <<EOF >"$SSL_CONF"
<VirtualHost *:443>
    ServerAdmin webmaster@$DOMAIN
    ServerName $DOMAIN

    DocumentRoot /var/www/html

    SSLEngine on
    SSLCertificateFile $CERT_FILE
    SSLCertificateKeyFile $KEY_FILE

    <FilesMatch "\.(cgi|shtml|phtml|php)$">
        SSLOptions +StdEnvVars
    </FilesMatch>

    <Directory /usr/lib/cgi-bin>
        SSLOptions +StdEnvVars
    </Directory>

    ErrorLog \${APACHE_LOG_DIR}/error.log
    CustomLog \${APACHE_LOG_DIR}/access.log combined
</VirtualHost>
EOF
fi

# Enable the SSL site configuration
echo "Enabling the default SSL site configuration..."
a2ensite default-ssl.conf || {
    echo "Failed to enable the SSL site configuration. Exiting."
    exit 1
}

# Restart Apache2
echo "Restarting Apache2..."
systemctl restart apache2 || {
    echo "Failed to restart Apache2. Exiting."
    exit 1
}

echo "Self-signed SSL certificate created and configured successfully!"
