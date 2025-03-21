#date: 2025-03-21T17:09:45Z
#url: https://api.github.com/gists/745c095b41606f6e0bbfa2dccb04a0d6
#owner: https://api.github.com/users/paranoia1st

#!/bin/bash

# Обновление системы
echo "Обновление системы..."
sudo apt update && sudo apt upgrade -y

# Установка необходимых пакетов
echo "Установка OpenVPN и Easy-RSA..."
sudo apt install openvpn easy-rsa iptables-persistent curl -y

# Автоопределение IPv4-адреса сервера
SERVER_IP=$(curl -s -4 https://api.ipify.org)
if [[ -z "$SERVER_IP" ]]; then
    echo "Не удалось определить IPv4-адрес сервера. Проверьте подключение к интернету."
    exit 1
fi

# Проверка формата IP-адреса
if [[ ! "$SERVER_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Получен некорректный IP-адрес: $SERVER_IP"
    exit 1
fi

echo "IPv4-адрес сервера: $SERVER_IP"

# Создание директории для PKI
echo "Настройка PKI..."
make-cadir ~/openvpn-ca
cd ~/openvpn-ca

# Настройка vars для Easy-RSA
cat <<EOF > vars
set_var EASYRSA_REQ_COUNTRY     "US"
set_var EASYRSA_REQ_PROVINCE    "California"
set_var EASYRSA_REQ_CITY        "San Francisco"
set_var EASYRSA_REQ_ORG         "My Company"
set_var EASYRSA_REQ_EMAIL       "admin@example.com"
set_var EASYRSA_REQ_OU          "My Organizational Unit"
EOF

# Инициализация PKI
./easyrsa init-pki

# Создание корневого сертификата (CA)
echo "Создание корневого сертификата (CA)..."
echo | ./easyrsa build-ca nopass

# Создание сертификата и ключа для сервера
echo "Создание сертификата и ключа для сервера..."
./easyrsa gen-req server nopass
echo | ./easyrsa sign-req server server

# Генерация Diffie-Hellman параметров
echo "Генерация DH-параметров..."
./easyrsa gen-dh

# Генерация ключа HMAC
echo "Генерация ключа HMAC..."
openvpn --genkey --secret ta.key

# Копирование файлов в директорию OpenVPN
echo "Копирование файлов в /etc/openvpn/server/..."
sudo cp pki/ca.crt pki/private/server.key pki/issued/server.crt pki/dh.pem ta.key /etc/openvpn/server/

# Создание конфигурационного файла сервера
echo "Создание конфигурации сервера..."
sudo tee /etc/openvpn/server/server.conf > /dev/null <<EOF
port 1194
proto udp
dev tun

ca ca.crt
cert server.crt
key server.key
dh dh.pem

server 10.8.0.0 255.255.255.0
ifconfig-pool-persist /var/log/openvpn/ipp.txt

push "redirect-gateway def1 bypass-dhcp"
push "dhcp-option DNS 8.8.8.8"
push "dhcp-option DNS 8.8.4.4"

keepalive 10 120
cipher AES-256-CBC
auth SHA256

tls-auth ta.key 0
topology subnet

user nobody
group nogroup

persist-key
persist-tun

status /var/log/openvpn/openvpn-status.log
verb 3

tun-mtu 1500
mssfix 1450
EOF

# Настройка IP-форвардинга
echo "Настройка IP-форвардинга..."
sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf
sudo sysctl -p

# Настройка iptables
echo "Настройка iptables..."
sudo iptables -t nat -A POSTROUTING -s 10.8.0.0/24 -o $(ip route | grep default | awk '{print $5}') -j MASQUERADE
sudo netfilter-persistent save

# Запуск и включение OpenVPN
echo "Запуск OpenVPN..."
sudo systemctl start openvpn-server@server
sudo systemctl enable openvpn-server@server

# Создание конфигурации клиента
echo "Создание конфигурации клиента..."
mkdir -p ~/client-configs/files
cp ~/openvpn-ca/pki/private/client1.key ~/client-configs/files/
cp ~/openvpn-ca/pki/issued/client1.crt ~/client-configs/files/
cp /etc/openvpn/server/ta.key ~/client-configs/files/

cat <<EOF > ~/client-configs/files/client.ovpn
client
dev tun
proto udp
remote $SERVER_IP 1194
resolv-retry infinite
nobind
persist-key
persist-tun
ca ca.crt
cert client1.crt
key client1.key
tls-auth ta.key 1
cipher AES-256-CBC
auth SHA256
verb 3
tun-mtu 1500
mssfix 1450
EOF

# Архивация файлов клиента
cd ~/client-configs/files
tar -czf client-config.tar.gz client.ovpn client1.crt client1.key ta.key

echo "Установка завершена!"
echo "Файл конфигурации клиента: ~/client-configs/files/client-config.tar.gz"