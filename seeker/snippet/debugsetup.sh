#date: 2024-09-09T17:03:15Z
#url: https://api.github.com/gists/334e1119b0b9a59f86fca7cedebab784
#owner: https://api.github.com/users/worksofliam

mkdir -p /QIBM/UserData/IBMiDebugService/certs 
chmod 755 /QIBM/UserData/IBMiDebugService/certs
rm -f /QIBM/UserData/IBMiDebugService/certs/debug_service.pfx /QIBM/UserData/IBMiDebugService/certs/debug_service.crt

export MY_JAVA_HOME="/QOpenSys/QIBM/ProdData/JavaVM/jdk11/64bit"

HNAME=$(hostname)
HNAMES=$(hostname -s)
# Dns lookup to get the IP address
IP=$(nslookup $(hostname) | grep Address | tail -n 1 | awk '{print $2}')

## Generate the certificate
pushd /QIBM/UserData/IBMiDebugService/certs
/QOpenSys/usr/bin/openssl genrsa -out debug_service.key 2048 
/QOpenSys/usr/bin/openssl req -new -key debug_service.key -out debug_service.csr -subj '/CN=$(HNAME)' 
/QOpenSys/usr/bin/openssl x509 -req -in debug_service.csr -signkey debug_service.key -out debug_service.crt -days 1095 -sha256 -req -extfile <(printf "subjectAltName=DNS:$HNAME,DNS:$HNAMES,IP:$IP") 
/QOpenSys/usr/bin/openssl pkcs12 -export -out debug_service.pfx -inkey debug_service.key -in debug_service.crt -password pass: "**********"
rm debug_service.key debug_service.csr
popd

## Set the password, password is random here. This is input to the encryptKeystorePassword.sh
export DEBUG_SERVICE_KEYSTORE_PASSWORD= "**********"

## Encrypt the password
EPW= "**********"

# Append required variables
echo "JAVA_HOME=/QOpenSys/QIBM/ProdData/JavaVM/jdk11/64bit" >> /QIBM/ProdData/IBMiDebugService/bin/DebugService.env
echo "DEBUG_SERVICE_KEYSTORE_FILE=/QIBM/UserData/IBMiDebugService/certs/debug_service.pfx" >> /QIBM/ProdData/IBMiDebugService/bin/DebugService.env
echo "DEBUG_SERVICE_KEYSTORE_PASSWORD= "**********"

## Start the service
system  "SBMJOB CMD(STRQSH CMD('/QOpenSys/pkgs/bin/bash -c /QIBM/ProdData/IBMiDebugService/bin/startDebugService.sh')) JOB(DBGSVCE) JOBQ(QSYS/QUSRNOMAX) JOBD(QSYS/QSYSJOBD) USER(*CURRENT)"  "SBMJOB CMD(STRQSH CMD('/QOpenSys/pkgs/bin/bash -c /QIBM/ProdData/IBMiDebugService/bin/startDebugService.sh')) JOB(DBGSVCE) JOBQ(QSYS/QUSRNOMAX) JOBD(QSYS/QSYSJOBD) USER(*CURRENT)"