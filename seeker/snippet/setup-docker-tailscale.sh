#date: 2025-06-05T17:08:56Z
#url: https://api.github.com/gists/45a1c21e075e83b03bad6ea9bfcbfac1
#owner: https://api.github.com/users/bob2187

#!/bin/bash

# Docker and Tailscale Container Setup Script
# Prerequisites: "**********"

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} Docker & Tailscale Setup${NC}"
    echo -e "${BLUE}================================${NC}"
    echo
}

# Function to prompt for input with default value
prompt_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " input
        eval "$var_name=\"\${input:-$default}\""
    else
        read -p "$prompt: " input
        eval "$var_name=\"$input\""
    fi
}

# Function to prompt for password (hidden input)
prompt_password() {
    local prompt="$1"
    local var_name="$2"
    
    read -s -p "$prompt: " input
    echo
    eval "$var_name=\"$input\""
}

# Function to prompt for yes/no
prompt_yes_no() {
    local prompt="$1"
    local default="$2"
    
    while true; do
        if [ "$default" = "y" ]; then
            read -p "$prompt [Y/n]: " yn
            yn=${yn:-y}
        else
            read -p "$prompt [y/N]: " yn
            yn=${yn:-n}
        fi
        
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check if running on Raspberry Pi
    if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
        print_warning "This doesn't appear to be a Raspberry Pi"
    fi
    
    # Check internet connectivity
    if ! ping -c 1 google.com &> /dev/null; then
        print_error "No internet connection detected"
        exit 1
    fi
    
    # Check if running as non-root user
    if [ "$EUID" -eq 0 ]; then
        print_error "Please don't run this script as root"
        exit 1
    fi
    
    print_status "System requirements check passed"
}

# Install Docker (following PiMyLifeUp guide)
install_docker() {
    print_status "Installing Docker using the official Docker installation script..."
    
    # Check if Docker is already installed
    if command -v docker &> /dev/null; then
        print_warning "Docker is already installed"
        docker --version
        return 0
    fi
    
    # Update system packages first
    print_status "Updating system packages..."
    sudo apt update
    sudo apt upgrade -y
    
    # Install Docker using the official script
    print_status "Downloading and running Docker installation script..."
    print_warning "This will download and execute the official Docker installation script"
    
    if prompt_yes_no "Proceed with Docker installation?" "y"; then
        curl -sSL https://get.docker.com | sh
    else
        print_error "Docker installation cancelled"
        exit 1
    fi
    
    # Add user to docker group
    print_status "Adding current user to docker group..."
    sudo usermod -aG docker $USER
    
    print_status "Docker installation completed"
    print_warning "You need to log out and log back in for Docker group permissions to take effect"
    
    # Check if user wants to reboot now or continue
    echo
    print_status "You have two options:"
    echo "1. Log out and log back in now (recommended)"
    echo "2. Continue with setup (will use sudo for Docker commands)"
    echo
    
    if prompt_yes_no "Log out now to apply Docker permissions?" "y"; then
        print_status "Please log back in and run this script again to continue with Tailscale setup"
        logout
    else
        print_warning "Continuing with setup using sudo for Docker commands"
        USE_SUDO_DOCKER=true
    fi
}

# Setup Docker Compose (if needed)
setup_docker_compose() {
    print_status "Checking Docker Compose availability..."
    
    # Check if docker compose plugin is available (comes with modern Docker)
    if docker compose version &> /dev/null 2>&1; then
        print_status "Docker Compose plugin is available"
        return 0
    elif [ "$USE_SUDO_DOCKER" = true ] && sudo docker compose version &> /dev/null 2>&1; then
        print_status "Docker Compose plugin is available (with sudo)"
        return 0
    fi
    
    # If we get here, Docker Compose might not be available
    print_warning "Docker Compose plugin not detected"
    print_status "The modern Docker installation should include Docker Compose plugin"
    print_status "If you encounter issues, you may need to install docker-compose separately"
}

# Test Docker installation
test_docker() {
    print_status "Testing Docker installation..."
    
    local docker_cmd="docker"
    if [ "$USE_SUDO_DOCKER" = true ]; then
        docker_cmd="sudo docker"
    fi
    
    # Run the hello-world container test
    print_status "Running Docker hello-world test..."
    if $docker_cmd run hello-world; then
        print_status "Docker test completed successfully!"
    else
        print_error "Docker test failed"
        exit 1
    fi
}
collect_tailscale_config() {
    print_status "Collecting Tailscale configuration..."
    echo
    
    prompt_input "Enter Tailscale Auth Key" "" TAILSCALE_AUTH_KEY
    
    if [ -z "$TAILSCALE_AUTH_KEY" ]; then
        print_error "Tailscale Auth Key is required"
        print_status "Get your auth key from: https://login.tailscale.com/admin/settings/keys"
        exit 1
    fi
    
    prompt_input "Enter Tailscale hostname" "$(hostname)" TAILSCALE_HOSTNAME
    prompt_input "Enter container name" "tailscale" CONTAINER_NAME
    
    # Additional Tailscale options
    echo
    print_status "Additional Tailscale options:"
    
    if prompt_yes_no "Enable subnet routing?" "n"; then
        prompt_input "Enter subnet to advertise (e.g., 192.168.1.0/24)" "" TAILSCALE_ROUTES
        ENABLE_ROUTES=true
    else
        ENABLE_ROUTES=false
    fi
    
    if prompt_yes_no "Enable exit node?" "n"; then
        ENABLE_EXIT_NODE=true
    else
        ENABLE_EXIT_NODE=false
    fi
    
    if prompt_yes_no "Accept routes from other nodes?" "y"; then
        ACCEPT_ROUTES=true
    else
        ACCEPT_ROUTES=false
    fi
}

# Configure system optimizations for Tailscale
configure_system_optimizations() {
    print_status "Configuring system optimizations for Tailscale..."
    
    # Enable IP forwarding
    print_status "Enabling IP forwarding..."
    
    # Check if already configured
    if ! grep -q "net.ipv4.ip_forward = 1" /etc/sysctl.conf; then
        echo 'net.ipv4.ip_forward = 1' | sudo tee -a /etc/sysctl.conf
        print_status "Added IPv4 forwarding to sysctl.conf"
    else
        print_status "IPv4 forwarding already configured"
    fi
    
    if ! grep -q "net.ipv6.conf.all.forwarding = 1" /etc/sysctl.conf; then
        echo 'net.ipv6.conf.all.forwarding = 1' | sudo tee -a /etc/sysctl.conf
        print_status "Added IPv6 forwarding to sysctl.conf"
    else
        print_status "IPv6 forwarding already configured"
    fi
    
    # Apply sysctl changes
    sudo sysctl -p
    
    # Configure UDP GRO optimizations
    print_status "Setting up UDP GRO optimizations..."
    
    # Create ethtool systemd service
    sudo tee /etc/systemd/system/ethtool-settings.service > /dev/null <<EOF
[Unit]
Description=Configure network interface optimizations
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 2
ExecStart=/sbin/ethtool -K eth0 rx-udp-gro-forwarding on rx-gro-list off
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    # Enable and start the service
    sudo systemctl daemon-reload
    sudo systemctl enable ethtool-settings
    sudo systemctl start ethtool-settings
    
    print_status "System optimizations configured successfully"
}
setup_tailscale_container() {
    print_status "Setting up Tailscale container..."
    
    local docker_cmd="docker"
    if [ "$USE_SUDO_DOCKER" = true ]; then
        docker_cmd="sudo docker"
    fi
    
    # Create directory for Tailscale data
    mkdir -p ~/tailscale-data
    
    # Build Docker run command
    DOCKER_CMD="$docker_cmd run -d \
        --name $CONTAINER_NAME \
        --hostname $TAILSCALE_HOSTNAME \
        --cap-add=NET_ADMIN \
        --cap-add=SYS_MODULE \
        --device /dev/net/tun \
        -v ~/tailscale-data:/var/lib/tailscale \
        -v /lib/modules:/lib/modules:ro \
        --restart unless-stopped"
    
    # Add environment variables
    DOCKER_CMD="$DOCKER_CMD -e TS_AUTH_KEY=$TAILSCALE_AUTH_KEY"
    DOCKER_CMD="$DOCKER_CMD -e TS_HOSTNAME=$TAILSCALE_HOSTNAME"
    
    # Add optional configurations
    TS_EXTRA_ARGS=""
    
    if [ "$ACCEPT_ROUTES" = true ]; then
        TS_EXTRA_ARGS="$TS_EXTRA_ARGS --accept-routes"
    fi
    
    if [ "$ENABLE_EXIT_NODE" = true ]; then
        TS_EXTRA_ARGS="$TS_EXTRA_ARGS --advertise-exit-node"
    fi
    
    if [ "$ENABLE_ROUTES" = true ] && [ -n "$TAILSCALE_ROUTES" ]; then
        TS_EXTRA_ARGS="$TS_EXTRA_ARGS --advertise-routes=$TAILSCALE_ROUTES"
    fi
    
    if [ -n "$TS_EXTRA_ARGS" ]; then
        DOCKER_CMD="$DOCKER_CMD -e TS_EXTRA_ARGS=\"$TS_EXTRA_ARGS\""
    fi
    
    # Add Tailscale image
    DOCKER_CMD="$DOCKER_CMD tailscale/tailscale:latest"
    
    print_status "Running Tailscale container..."
    eval $DOCKER_CMD
    
    # Wait for container to start
    sleep 5
    
    # Check container status
    if $docker_cmd ps | grep -q $CONTAINER_NAME; then
        print_status "Tailscale container started successfully"
        $docker_cmd logs $CONTAINER_NAME
    else
        print_error "Failed to start Tailscale container"
        $docker_cmd logs $CONTAINER_NAME
        exit 1
    fi
}

# Create Docker Compose file (based on your specific configuration)
create_docker_compose() {
    print_status "Creating Docker Compose file..."
    
    # Create directory structure
    mkdir -p "$TAILSCALE_DIR"
    mkdir -p "$STATE_DIR"
    
    # Build the tailscale up command with options
    TAILSCALE_UP_CMD="tailscale up --authkey=\$TS_AUTH_KEY --hostname=$TAILSCALE_HOSTNAME"
    
    if [ "$ENABLE_EXIT_NODE" = true ]; then
        TAILSCALE_UP_CMD="$TAILSCALE_UP_CMD --advertise-exit-node"
    fi
    
    if [ -n "$TAILSCALE_ROUTES" ]; then
        TAILSCALE_UP_CMD="$TAILSCALE_UP_CMD --advertise-routes=$TAILSCALE_ROUTES"
    fi
    
    if [ "$ACCEPT_ROUTES" = true ]; then
        TAILSCALE_UP_CMD="$TAILSCALE_UP_CMD --accept-routes"
    fi
    
    # Create docker-compose.yml
    cat > "$TAILSCALE_DIR/docker-compose.yml" <<EOF
services:
  tailscale:
    image: tailscale/tailscale:stable
    container_name: $CONTAINER_NAME
    hostname: $TAILSCALE_HOSTNAME
    restart: unless-stopped
    network_mode: host
    cap_add:
      - NET_ADMIN
      - NET_RAW
    volumes:
      - $STATE_DIR:/var/lib/tailscale
      - /dev/net/tun:/dev/net/tun
    environment:
      - TS_STATE_DIR=/var/lib/tailscale
      - TS_AUTH_KEY=$TAILSCALE_AUTH_KEY
      - TS_ACCEPT_DNS=$ACCEPT_DNS
    command: |
      sh -c "
      # Start tailscaled and wait for it to initialize
      tailscaled &
      sleep 5
      
      # Configure and start tailscale with all options
      $TAILSCALE_UP_CMD
      
      # Keep container running
      tail -f /dev/null
      "
EOF

    print_status "Docker Compose file created at $TAILSCALE_DIR/docker-compose.yml"
    
    # Create helpful script files
    create_helper_scripts
}

# Main setup function
main_setup() {
    print_header
    
    # Check requirements
    check_requirements
    
    # Install Docker using PiMyLifeUp method
    install_docker
    
    # Test Docker installation
    test_docker
    
    # Setup Docker Compose
    setup_docker_compose
    
    # Configure system optimizations
    configure_system_optimizations
    
    # Collect Tailscale configuration
    collect_tailscale_config
    
    # Confirmation
    echo
    print_status "Configuration Summary:"
    echo "Container name: $CONTAINER_NAME"
    echo "Tailscale hostname: $TAILSCALE_HOSTNAME"
    echo "Docker directory: $TAILSCALE_DIR"
    echo "State directory: $STATE_DIR"
    echo "Auth key: [CONFIGURED]"
    echo "Advertised routes: $TAILSCALE_ROUTES"
    echo "Enable exit node: $ENABLE_EXIT_NODE"
    echo "Accept routes: $ACCEPT_ROUTES"
    echo "Accept DNS: $ACCEPT_DNS"
    echo
    
    if ! prompt_yes_no "Proceed with Tailscale container setup?" "y"; then
        print_warning "Setup cancelled by user"
        exit 0
    fi
    
    # Create Docker Compose configuration
    create_docker_compose
    
    # Start Tailscale container
    start_tailscale_container
    
# Start Tailscale container
start_tailscale_container() {
    print_status "Starting Tailscale container..."
    
    cd "$TAILSCALE_DIR"
    
    local docker_cmd="docker compose"
    if [ "$USE_SUDO_DOCKER" = true ]; then
        docker_cmd="sudo docker compose"
    fi
    
    # Start the container
    $docker_cmd up -d
    
    # Wait for container to start and authenticate
    print_status "Waiting for Tailscale to authenticate and connect..."
    sleep 15
    
    # Check container status
    if docker ps | grep -q $CONTAINER_NAME; then
        print_status "Tailscale container started successfully"
        print_status "Checking authentication status..."
        
        # Check if Tailscale is authenticated
        sleep 5
        if docker exec $CONTAINER_NAME tailscale status > /dev/null 2>&1; then
            print_status "✅ Tailscale authenticated successfully!"
            docker exec $CONTAINER_NAME tailscale status
        else
            print_warning "⚠️  Tailscale may still be connecting..."
            print_status "Container logs:"
            docker logs $CONTAINER_NAME --tail 20
        fi
        
        echo
        print_status "Management Information:"
        print_status "- Check status: cd $TAILSCALE_DIR && ./status.sh"
        print_status "- View logs: cd $TAILSCALE_DIR && ./logs.sh"
        print_status "- Tailscale admin: https://login.tailscale.com/admin/machines"
        
    else
        print_error "Failed to start Tailscale container"
        docker logs $CONTAINER_NAME
        exit 1
    fi
}
    
    print_status "Setup completed successfully!"
    echo
    print_status "🎉 Tailscale should now be fully connected and authenticated!"
    echo
    print_status "Next Steps:"
    echo "1. Navigate to your Tailscale directory: cd $TAILSCALE_DIR"
    echo "2. Use the helper scripts to manage your container:"
    echo "   - ./start.sh    - Start Tailscale"
    echo "   - ./stop.sh     - Stop Tailscale"
    echo "   - ./logs.sh     - View logs"
    echo "   - ./status.sh   - Check status"
    echo "   - ./restart.sh  - Restart container"
    echo "3. Visit https://login.tailscale.com/admin/machines to manage your node"
    echo "4. Approve subnet routes and exit node capabilities if needed"
    echo
    print_status "Your Tailscale node should now be visible and active in your admin panel"
    print_status "System optimizations (IP forwarding and UDP GRO) have been configured"
    echo
    print_warning "💡 Important: Save your auth key securely - you'll need it for future reinstalls"
}

# Run main setup
main_setupr future reinstalls"
}

# Run main setup
main_setup