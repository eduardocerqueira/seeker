#date: 2023-05-22T17:00:00Z
#url: https://api.github.com/gists/3b717e93df420a403be93062e884152e
#owner: https://api.github.com/users/schinwald

# Create a symbolic link between this service and the /etc/systemd/system/ service
sudo systemctl link docker-compose-application.service

# Enable the service to run on startup
sudo systemctl enable docker-compose-application.service

# Test the service to make sure it is working
sudo systemctl start docker-compose-application.service
sudo systemctl status docker-compose-application.service
sudo systemctl daemon-reload