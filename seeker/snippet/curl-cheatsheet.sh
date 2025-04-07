#date: 2025-04-07T16:50:49Z
#url: https://api.github.com/gists/617e8501af8f3c3a46f327016d31f3d4
#owner: https://api.github.com/users/nayandas69

# Basic GET request
curl https://example.com

# GET request with custom headers
curl -H "Authorization: "**********"://api.example.com/data

# Follow redirects
curl -L https://example.com

# Save output to a file
curl -o filename.html https://example.com

# Download with a specific filename
curl -O https://example.com/file.zip

# Send POST request with form data
curl -X POST -d "username=user&password=pass" https: "**********"

# Send POST request with JSON data
curl -X POST -H "Content-Type: application/json" \
-d '{"name":"John", "age":30}' https://api.example.com/users

# Add multiple headers
curl -H "Accept: "**********": Bearer <token>" https://api.example.com/data

# Upload a file
curl -F "file=@/path/to/file.txt" https://example.com/upload

# PUT request with JSON
curl -X PUT -H "Content-Type: application/json" \
-d '{"title":"Updated Title"}' https://api.example.com/post/1

# DELETE request
curl -X DELETE https://api.example.com/item/123

# Show response headers only
curl -I https://example.com

# Show request and response details (debug)
curl -v https://example.com

# Set timeout (10 seconds)
curl --max-time 10 https://example.com

# Use a proxy
curl -x http://proxy.example.com:8080 https://example.com

# Basic authentication
curl -u username: "**********"://example.com

# Use a specific HTTP method
curl -X PATCH -d "data" https://api.example.com/resource/1

# Send custom User-Agent
curl -A "MyUserAgent/1.0" https://example.com

# Send cookies
curl -b "name=value" https://example.com

# Save cookies to a file
curl -c cookies.txt https://example.com

# Load cookies from a file
curl -b cookies.txt https://example.com

# Limit download speed
curl --limit-rate 100K https://example.com

# Show progress bar
curl -# https://example.com

# Retry on failure
curl --retry 5 https://example.com

# Use a different HTTP version
curl --http2 https://example.com

# Send a custom request method
curl -X OPTIONS https://example.com

# Use a different port
curl -P 8080 https://example.com

# Send a request with a specific protocol
curl --http1.1 https://example.com

# Use a specific CA certificate
curl --cacert /path/to/certificate.pem https://example.com

# Use a specific client certificate
curl --cert /path/to/client.crt --key /path/to/client.key https://example.com

# Use a specific TLS version
curl --tlsv1.2 https://example.com

# Use a specific cipher
curl --ciphers 'ECDHE-RSA-AES256-GCM-SHA384' https://example.com
AES256-GCM-SHA384' https://example.com
