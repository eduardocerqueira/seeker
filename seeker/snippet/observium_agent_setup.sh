#date: 2021-08-31T13:05:52Z
#url: https://api.github.com/gists/ab2a1460d293d8504c3cc381a04706e3
#owner: https://api.github.com/users/flamingsnowflake

# CONFIG
SNMP_COMMUNITY=public
SYSCONTACT=dan
SYSLOCATION=Berlin

# get packages
apt-get install snmpd xinetd vim

mkdir -p /opt/observium && cd /opt

wget http://www.observium.org/observium-community-latest.tar.gz
tar zxvf observium-community-latest.tar.gz

sed -e "/SNMPDOPTS=/ s/^#*/SNMPDOPTS='-Lsd -Lf \/dev\/null -u snmp -p \/var\/run\/snmpd.pid'\n#/" -i /etc/default/snmpd

wget http://www.observium.org/svn/observer/trunk/scripts/distro
mv distro /usr/bin/distro
chmod 755 /usr/bin/distro

mv /etc/snmp/snmpd.conf /etc/snmp/snmpd.conf.org
cat >/etc/snmp/snmpd.conf <<EOL
com2sec readonly  default         $SNMP_COMMUNITY
group MyROGroup v1         readonly
group MyROGroup v2c        readonly
group MyROGroup usm        readonly
view all    included  .1                               80
access MyROGroup ""      any       noauth    exact  all    none   none
syslocation $SYSLOCATION
syscontact $SYSCONTACT
#This line allows Observium to detect the host OS if the distro script is installed
extend .1.3.6.1.4.1.2021.7890.1 distro /usr/bin/distro 
EOL


cp observium/scripts/observium_agent_xinetd /etc/xinetd.d/observium_agent

# change "only_from" to match your observium server ip
vim /etc/xinetd.d/observium_agent

cp observium/scripts/observium_agent /usr/bin/observium_agent

mkdir -p /usr/lib/observium_agent/local

# copy all agent helper scripts that you want (bind, dpkg, freeradius, ksm, munin, mysql.cnf, ntpd, postfix_qshape, powerdns, rpm, shoutcast.conf, unbound, apache, crashplan, drbd, hddtemp, lmsensors, munin-scripts, nfs, nvidia-smi, postgresql.conf, powerdns-recursor, sabnzbd-qstatus, shoutcast.default.conf, vmwaretools, asterisk, dmi, exim-mailqueue.sh, ipmitool-sensor, memcached, mysql, nginx, postfix_mailgraph, postgresql.pl, raspberrypi, shoutcast, temperature)
# to make this work you have to enable the unix-agent module on your device's settings or globally in config.php ($config['poller_modules']['unix-agent'] = 1;)
cp observium/scripts/agent-local/AGENTHELPER /usr/lib/observium_agent/local/

/etc/init.d/xinetd restart
/etc/init.d/snmpd restart 
