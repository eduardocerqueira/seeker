#date: 2026-03-03T17:33:36Z
#url: https://api.github.com/gists/57346af62d5af5b39a7ac62098479faf
#owner: https://api.github.com/users/emergentcomplex

#!/bin/bash
# =================================================================
# AlmaLinux 10 STIG GUI Hardening & Verification Script
# Purpose: Apply DISA STIG compliance to a GUI-enabled node.
# Note: Manually expunges openssl-pkcs11 due to repo unavailability.
# =================================================================

set -e

# 1. Environment Setup
echo "[*] Installing EPEL and OpenSCAP Tools..."
sudo dnf install epel-release -y
sudo dnf install openscap-scanner scap-security-guide ansible-core -y
sudo dnf makecache

# 2. Global Collection Install (Required for Sudo/Root execution)
echo "[*] Installing Ansible Collections..."
sudo ansible-galaxy collection install ansible.posix community.general

# 3. Playbook Generation
echo "[*] Generating Playbook for STIG GUI Profile..."
DATALOCATION="/usr/share/xml/scap/ssg/content/ssg-almalinux10-ds.xml"
PROFILE="xccdf_org.ssgproject.content_profile_stig_gui"
PLAYBOOK="alma10-gui-stig.yml"

oscap xccdf generate fix --profile "$PROFILE" \
--fix-type ansible \
--output "$PLAYBOOK" \
"$DATALOCATION"

# 4. Surgical Strike: Expunge openssl-pkcs11 (Lines 12327-12342)
# We delete these specific lines to prevent the "No package available" failure.
echo "[*] Removing non-existent package references (Sanity Fix)..."
sed -i '12327,12342d' "$PLAYBOOK"

# 5. Execute Hardening
echo "[*] Executing Playbook (This may take several minutes)..."
sudo ansible-playbook -i "localhost," -c local "$PLAYBOOK"

# 6. Final Audit Report
echo "[*] Generating Post-Hardening Compliance Report..."
REPORT_NAME="stig_audit_$(date +%F).html"
sudo oscap xccdf eval --profile "$PROFILE" \
--results /root/results.xml \
--report "/root/$REPORT_NAME" \
"$DATALOCATION" || true

echo "-------------------------------------------------------"
echo "DONE. System hardened (minus missing openssl-pkcs11)."
echo "Audit Report Location: /root/$REPORT_NAME"
echo "ACTION REQUIRED: Reboot to apply all security policies."
echo "-------------------------------------------------------"
