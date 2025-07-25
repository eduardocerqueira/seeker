#date: 2025-07-25T16:56:10Z
#url: https://api.github.com/gists/c7b90995deb348358053c61c56831fd3
#owner: https://api.github.com/users/Mr-Bossman

#!/bin/bash

do_install() {
cat <<EOF > /etc/systemd/system/ronginx.service
[Unit]
Description=Read-only filesystem support for Nginx
After=tmp.mount

[Service]
Type=oneshot
ExecStart=/bin/ronginx run

[Install]
WantedBy=multi-user.target
RequiredBy=nginx.service
EOF

mv "$0" /bin/ronginx
systemctl enable ronginx.service
}

run() {
	chown -R www-data:www-data /var/lib/nginx
}

help() {
	echo To allow Nginx to run on read-only filesystem run:
	echo "$0" run
	echo To install this service run:
	echo "$0" install

}

case "$1" in
	"install")
		do_install
	;;

	"run")
		run
	;;

	*)
		help
	;;

esac
