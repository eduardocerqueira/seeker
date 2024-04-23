#date: 2024-04-23T17:10:11Z
#url: https://api.github.com/gists/526c06fedc3ca87934d07a3d80f86b63
#owner: https://api.github.com/users/mik0l

su -lc '
  mkdir -p /etc/security/access.d
  echo "-:ALL:ALL EXCEPT LOCAL" > /etc/security/access.d/local.conf
  authselect enable-feature with-pamaccess
  systemctl --now disable abrtd auditd avahi-daemon chronyd cups dnf-makecache.timer pcscd rsyslog
  rm -rf /var/log/journal
  ln -s /dev/null /etc/sysctl.d/50-coredump.conf
'
