#date: 2025-12-29T17:02:27Z
#url: https://api.github.com/gists/88dd9ed211b65801e87d3af5b081cb11
#owner: https://api.github.com/users/coleleavitt

#!/bin/bash
#
# ThinkPad P16 Gen 3 - GPE 0x6E Interrupt Storm Fix
# ACPI Table Override Installer for Gentoo Linux
#
# This script installs a fixed SSDT17 table that prevents the GPE 0x6E
# interrupt storm caused by a firmware bug in the Thunderbolt/GPU hotplug handler.
#
# The fix unconditionally clears all GPIO sources at the end of the PL6E handler,
# preventing spurious re-triggering.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root"
fi

# Source directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AML_FILE="${SCRIPT_DIR}/ssdt17-fixed.aml"
CPIO_FILE="${SCRIPT_DIR}/acpi-override.cpio"

# Destination
ACPI_DIR="/lib/firmware/acpi-override"
FINAL_CPIO="/lib/firmware/acpi-override.cpio"

# Check for required files
if [[ ! -f "$AML_FILE" ]]; then
    error "Cannot find ssdt17-fixed.aml in ${SCRIPT_DIR}"
fi

info "ThinkPad P16 Gen 3 - GPE 0x6E Fix Installer"
info "============================================"
echo ""

# Step 1: Create directory structure and copy AML
info "Step 1: Installing ACPI override table..."
mkdir -p "${ACPI_DIR}/kernel/firmware/acpi"
cp "$AML_FILE" "${ACPI_DIR}/kernel/firmware/acpi/"
chmod 644 "${ACPI_DIR}/kernel/firmware/acpi/ssdt17-fixed.aml"
info "  -> Installed to ${ACPI_DIR}/kernel/firmware/acpi/ssdt17-fixed.aml"

# Step 2: Create CPIO archive
info "Step 2: Creating CPIO archive..."
cd "${ACPI_DIR}"
find . -print0 | cpio --null --create --format=newc > "${FINAL_CPIO}" 2>/dev/null
chmod 644 "${FINAL_CPIO}"
info "  -> Created ${FINAL_CPIO}"

# Step 3: Detect initramfs generator
info "Step 3: Detecting initramfs generator..."

DRACUT_CONF="/etc/dracut.conf.d/acpi-override.conf"
GENKERNEL_CONF="/etc/genkernel.conf"

if command -v dracut &>/dev/null; then
    info "  -> Found dracut"
    
    # Create dracut configuration
    cat > "${DRACUT_CONF}" << 'EOF'
# ThinkPad P16 Gen 3 - GPE 0x6E Fix
# Include ACPI override table in initramfs
acpi_override="yes"
acpi_table_dir="/lib/firmware/acpi-override/kernel/firmware/acpi"
EOF
    
    info "  -> Created ${DRACUT_CONF}"
    
    # Regenerate initramfs
    info "Step 4: Regenerating initramfs with dracut..."
    KERNEL_VERSION=$(uname -r)
    dracut --force "/boot/initramfs-${KERNEL_VERSION}.img" "${KERNEL_VERSION}"
    info "  -> Regenerated /boot/initramfs-${KERNEL_VERSION}.img"
    
elif command -v genkernel &>/dev/null; then
    info "  -> Found genkernel"
    warn "  -> genkernel requires manual configuration"
    echo ""
    echo "Add to /etc/genkernel.conf:"
    echo "  FIRMWARE_DIR=\"/lib/firmware/acpi-override\""
    echo ""
    echo "Then regenerate initramfs:"
    echo "  genkernel --install initramfs"
    echo ""
    
else
    # Manual method - prepend CPIO to existing initramfs
    info "  -> No dracut or genkernel found, using manual method"
    
    # Find current initramfs
    KERNEL_VERSION=$(uname -r)
    INITRAMFS=""
    
    for path in "/boot/initramfs-${KERNEL_VERSION}.img" \
                "/boot/initramfs-${KERNEL_VERSION}" \
                "/boot/initrd-${KERNEL_VERSION}" \
                "/boot/initrd.img-${KERNEL_VERSION}"; do
        if [[ -f "$path" ]]; then
            INITRAMFS="$path"
            break
        fi
    done
    
    if [[ -z "$INITRAMFS" ]]; then
        warn "Could not find initramfs for kernel ${KERNEL_VERSION}"
        echo ""
        echo "Manual installation required:"
        echo "  1. Find your initramfs file"
        echo "  2. Prepend the ACPI override CPIO:"
        echo "     cat ${FINAL_CPIO} /path/to/initramfs > /path/to/initramfs.new"
        echo "     mv /path/to/initramfs.new /path/to/initramfs"
        echo ""
    else
        info "Step 4: Prepending ACPI override to initramfs..."
        BACKUP="${INITRAMFS}.backup-$(date +%Y%m%d%H%M%S)"
        cp "$INITRAMFS" "$BACKUP"
        info "  -> Backed up to ${BACKUP}"
        
        cat "${FINAL_CPIO}" "$BACKUP" > "$INITRAMFS"
        info "  -> Updated ${INITRAMFS}"
    fi
fi

# Step 5: Verify kernel config
info "Step 5: Checking kernel configuration..."
if [[ -f /boot/config-$(uname -r) ]]; then
    if grep -q "CONFIG_ACPI_TABLE_UPGRADE=y" /boot/config-$(uname -r); then
        info "  -> CONFIG_ACPI_TABLE_UPGRADE=y (good)"
    else
        warn "  -> CONFIG_ACPI_TABLE_UPGRADE not enabled!"
        echo "     You may need to enable this in your kernel config and rebuild"
    fi
else
    warn "  -> Could not verify kernel config"
fi

echo ""
info "============================================"
info "Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Reboot your system"
echo "  2. Verify the fix with:"
echo "     dmesg | grep -i 'ACPI.*SSDT.*PchGpe'"
echo "     cat /sys/firmware/acpi/interrupts/gpe6E"
echo ""
echo "The interrupt count should stay low (not increasing rapidly)"
echo ""
echo "To verify the override loaded, check dmesg for:"
echo "  'ACPI: Table Upgrade: install [SSDT-LENOVO-PchGpe ]'"
echo ""
