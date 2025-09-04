#date: 2025-09-04T17:02:46Z
#url: https://api.github.com/gists/5f0534ea25bab2e91616e3456450b962
#owner: https://api.github.com/users/IBruteDude

# -------------------------------
# Install latest ngrok
# -------------------------------
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok

# -------------------------------
# Authenticate ngrok
# -------------------------------
ngrok config add-authtoken $AUTH_TOKEN

# -------------------------------
# Install latest code-server
# -------------------------------
echo "Installing code-server..."
curl -fsSL https://code-server.dev/install.sh | sh

# -------------------------------
# Create config directory
# -------------------------------
mkdir -p ~/.config/code-server
cat <<EOF > ~/.config/code-server/config.yaml
bind-addr: 127.0.0.1:$PORT
auth: "**********"
password: "**********"
cert: false
EOF

# -------------------------------
# Launch code-server & ngrok tunnel
# -------------------------------
echo "Starting code-server on port $PORT..."
nohup code-server /content/drive/MyDrive/workspace &

echo "Starting ngrok tunnel"
nohup ngrok http $PORT &
