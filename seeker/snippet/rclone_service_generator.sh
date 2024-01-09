#date: 2024-01-09T17:06:14Z
#url: https://api.github.com/gists/a813902ba3cdf286f6d6e4ec5f703b2b
#owner: https://api.github.com/users/m4rc377

#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo"
    exit
fi

get_user() {
    user="${SUDO_USER:-${USER}}"
    echo $user
}

get_homeuser() {
    homeuser="/home/$(get_user)"
    echo $homeuser
}

user=$(get_user)
userhome=$(get_homeuser)

#config
generated_location="."
generated_system_location="/etc/systemd/system"

# Remote Name (Required)
while true; do
    read -p "Remote Name in config : " remote_name
    if [[ -z "$remote_name" || "$remote_name" =~ ^[[:space:]]+$ || "$remote_name" =~ [[:space:]] ]]; then
        echo 'Remote Name is Required...!' 
    else
        break
    fi
done

# Service Name
read -p "Service Name (\"$remote_name\"): " service_name
if [[ -z "$service_name" ]]; then
    service_name=$remote_name
fi

# (optional)
read -p "Service Description (\"\"): " service_desc

# optional
read -p "OS username (\"$user\"): " os_username
    if [[ -z "$os_username" ]]; then
        os_username=$user
    fi

# required
while true; do
    read -p "Mounting Dir Name (\"$service_name\"): " dir_name
#    if [[ -z "$dir_name" || "$dir_name" =~ ^[[:space:]]+$ || "$dir_name" =~ [[:space:]] ]]; then
#        echo 'Mount Point is Required...!' 
#    else
#        break
#    fi
    if [[ -z "$dir_name" ]]; then
        dir_name=$service_name
        break
    elif [[ "$dir_name" =~ ^[[:space:]]+$ || "$dir_name" =~ [[:space:]] ]]; then
        echo 'Mount Point is Required...!' 
    else
        echo 'Something wrong with mount point...!' 
        exit 1;
    fi


done

read -p "Mounting Point ("/media/$user/$dir_name"): " mount_point

read -p "rclone Config Location ("/home/$user/.config/rclone/rclone.conf"): " rclone_config
if [[ -z "$rclone_config" ]]; then
    rclone_config="/home/$USER/.config/rclone/rclone.conf"
fi

### Service Template
srvc=("[Unit]
Description=\"${service_desc}\"
After=network-online.target
Wants=network-online.target

[Service]
User=$os_username
Group=1000
TimeoutStartSec=6000

Type=notify
ExecStartPre=+/bin/mkdir -p /media/$os_username/$dir_name
ExecStartPre=+/bin/chown -hR $os_username:1000 /media/$os_username/$dir_name
ExecStart=/usr/bin/rclone mount --vfs-cache-mode writes $remote_name: /media/$os_username/$dir_name
ExecStop=/bin/fusermount -uz /media/$os_username/$dir_name
ExecStopPost=+/bin/rm -r /media/$os_username/$dir_name
Restart=on-abort
RestartSec=10

[Install]
WantedBy=multi-user.target
" )

### Service Generator Function
# Usage : generating_service {service} destination
generating_service() {
    local service=${1}
    local destination=${2:-"$generated_location"}
    echo " dest: $2"
    echo " dest: $destination"
    if [[ -z "$service" ]]; then
        exit 1;
    fi
    echo "File generated"
    cat <<-EOF >> $destination
$service
EOF
}


### SInstall COnfirmation
read -r -p "Do you want to installing to System !!!? [y/N] " install
if [[ -z "$install" ]]; then
    install="no"
fi

case "$install" in
    [yY][eE][sS]|[yY]) 
        service_dest="$generated_system_location/$service_name.service"
        generating_service "$srvc" "$service_dest"
        # Enable & running the service 
        systemctl enable $service_name
        systemctl start $service_name
        ;;
    [nN][oO]) 
        service_dest="$generated_location/$service_name.service"
        generating_service "$srvc" "$service_dest"
        ;;
    *)
        echo "Abort ...."
        ;;
esac

echo "Service Generated"
echo "file generate in >" $service_dest


# sudo systemctl stop box-test 
# sudo systemctl disable box-test