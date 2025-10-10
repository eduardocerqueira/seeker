#date: 2025-10-10T17:02:44Z
#url: https://api.github.com/gists/c3d37dd00c7f0385e40f68c0a91b8370
#owner: https://api.github.com/users/repositorioinformatico

#!/bin/bash

# ========================================
# SCRIPT DE LABORATORIO: SERVIDOR DHCP CON PODMAN
# Arquitectura: Servidor (2 interfaces: NAT + red interna)
#               Cliente (1 interfaz: solo red interna, sin internet)
# ========================================

# Limpiar todo
echo "Limpiando contenedores, imágenes y redes existentes..."
podman stop -a 2>/dev/null
podman rm -a 2>/dev/null
podman rmi -a 2>/dev/null
podman network prune -f

# ========================================
# Crear imagen base con herramientas
# ========================================

echo "Creando imagen base con herramientas de red..."
podman run -d --name temp-base ubuntu:22.04 sleep infinity
podman exec temp-base apt update
podman exec temp-base apt install -y vim iputils-ping isc-dhcp-client isc-dhcp-server iproute2 net-tools
podman exec temp-base bash -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf"
podman exec temp-base bash -c "echo 'nameserver 8.8.4.4' >> /etc/resolv.conf"

# Hacer commit de la imagen base
echo "Guardando imagen base..."
podman commit temp-base ubuntu-base-tools:1
podman stop temp-base
podman rm temp-base

# ========================================
# Crear red interna (sin internet)
# ========================================

echo "Creando red interna sin salida a internet..."
podman network create --internal red-interna

# ========================================
# Crear contenedor SERVIDOR (2 interfaces: NAT + red interna)
# ========================================

echo "Creando contenedor servidor..."
podman run -dit --name servidor --cap-add=NET_RAW --network podman ubuntu-base-tools:1 bash
podman network connect red-interna servidor
podman exec servidor bash -c "echo 'nameserver 8.8.8.8' > /etc/resolv.conf && echo 'nameserver 8.8.4.4' >> /etc/resolv.conf"

# Obtener IP de eth1 del servidor
SERVIDOR_ETH1_IP=$(podman exec servidor ip -4 addr show eth1 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
echo "IP del servidor en eth1: $SERVIDOR_ETH1_IP"

# ========================================
# Crear contenedor CLIENTE (solo red interna, sin internet)
# ========================================

echo "Creando contenedor cliente..."
podman run -dit --name cliente --cap-add=NET_RAW --network red-interna ubuntu-base-tools:1 bash

# Obtener MAC del cliente
CLIENTE_MAC=$(podman exec cliente ip addr show eth0 | grep "link/ether" | awk '{print $2}')
echo "MAC del cliente: $CLIENTE_MAC"

# IMPORTANTE: Liberar la IP estática que Podman asignó automáticamente
echo "Liberando IP estática automática del cliente..."
podman exec cliente ip addr flush dev eth0
podman exec cliente ip link set eth0 down
podman exec cliente ip link set eth0 up

# ========================================
# Configurar servidor DHCP
# ========================================

echo "Configurando servidor DHCP..."

# Configurar interfaz para DHCP
podman exec servidor bash -c "cat > /etc/default/isc-dhcp-server << 'EOF'
INTERFACESv4=\"eth1\"
INTERFACESv6=\"\"
EOF"

# Configurar dhcpd.conf
podman exec servidor bash -c "cat > /etc/dhcp/dhcpd.conf << 'EOF'
default-lease-time 600;
max-lease-time 7200;
authoritative;

subnet 10.89.0.0 netmask 255.255.255.0 {
  range 10.89.0.10 10.89.0.50;
  option routers SERVIDOR_IP_PLACEHOLDER;
  option domain-name-servers 8.8.8.8;
  option subnet-mask 255.255.255.0;
}

host cliente {
  hardware ethernet CLIENTE_MAC_PLACEHOLDER;
  fixed-address 10.89.0.100;
}
EOF"

# Reemplazar placeholders con valores reales
podman exec servidor sed -i "s/SERVIDOR_IP_PLACEHOLDER/$SERVIDOR_ETH1_IP/g" /etc/dhcp/dhcpd.conf
podman exec servidor sed -i "s/CLIENTE_MAC_PLACEHOLDER/$CLIENTE_MAC/g" /etc/dhcp/dhcpd.conf

# Crear archivo de leases
podman exec servidor touch /var/lib/dhcp/dhcpd.leases

# Arrancar servidor DHCP en background
echo "Arrancando servidor DHCP..."
podman exec -d servidor /usr/sbin/dhcpd eth1

echo "Esperando 3 segundos para que el servidor DHCP esté listo..."
sleep 3

# Verificar que el servidor DHCP está corriendo
echo "Verificando que el servidor DHCP está corriendo:"
podman exec servidor ps aux | grep dhcpd | grep -v grep

# ========================================
# Cliente solicita IP por DHCP
# ========================================

echo ""
echo "Cliente solicitando IP por DHCP..."
podman exec cliente dhclient -v eth0

# Esperar un momento para que se complete la asignación
sleep 2

# Verificar IP asignada al cliente
echo ""
echo "=========================================="
echo "IP asignada al cliente:"
podman exec cliente ip addr show eth0 | grep "inet "
echo "=========================================="

# ========================================
# Verificar concesión de IP en el servidor
# ========================================

echo ""
echo "=========================================="
echo "Leases concedidos por el servidor:"
podman exec servidor cat /var/lib/dhcp/dhcpd.leases
echo "=========================================="

# ========================================
# Verificar que cliente NO tiene internet
# ========================================

echo ""
echo "=========================================="
echo "Verificando que el cliente NO tiene internet:"
podman exec cliente ping -c 2 8.8.8.8 2>&1 | grep -E "(Network is unreachable|100% packet loss)" && echo "✓ Cliente correctamente SIN internet" || echo "✗ PROBLEMA: Cliente tiene internet"
echo "=========================================="

# ========================================
# Probar conectividad en red interna
# ========================================

echo ""
echo "=========================================="
echo "Ping desde cliente al servidor ($SERVIDOR_ETH1_IP):"
podman exec cliente ping -c 3 $SERVIDOR_ETH1_IP
echo "=========================================="

echo ""
echo "=========================================="
echo "Ping desde servidor al cliente (10.89.0.100):"
podman exec servidor ping -c 3 10.89.0.100
echo "=========================================="

# ========================================
# Resumen final
# ========================================

echo ""
echo "========================================"
echo "LABORATORIO COMPLETADO EXITOSAMENTE"
echo "========================================"
echo ""
echo "ARQUITECTURA:"
echo "  SERVIDOR:"
echo "    - eth0: Internet (NAT)"
echo "    - eth1: $SERVIDOR_ETH1_IP (red interna)"
echo "    - Servidor DHCP corriendo en eth1"
echo ""
echo "  CLIENTE:"
echo "    - eth0: 10.89.0.100 (red interna, asignada por DHCP)"
echo "    - SIN acceso a internet"
echo "    - IP reservada por MAC: $CLIENTE_MAC"
echo ""
echo "COMANDOS ÚTILES:"
echo "  - Entrar al servidor: podman exec -it servidor bash"
echo "  - Entrar al cliente: podman exec -it cliente bash"
echo "  - Ver leases: podman exec servidor cat /var/lib/dhcp/dhcpd.leases"
echo "  - Ver logs DHCP: podman exec servidor journalctl -u isc-dhcp-server"
echo "  - Renovar IP cliente: podman exec cliente dhclient -r eth0 && podman exec cliente dhclient eth0"
echo "========================================"