#!/usr/bin/env bash
set -e

echo "=============================="
echo "Microsoft Defender hardening"
echo "=============================="

echo "[1/7] Updating OS packages..."
sudo apt-get update
sudo apt-get -y full-upgrade
sudo apt-get -y autoremove --purge

echo
echo "[2/7] Disabling Microsoft Defender passive mode..."
sudo mdatp config passive-mode --value disabled || true

echo
echo "[3/7] Enabling real-time protection..."
sudo mdatp config real-time-protection --value enabled || true

echo
echo "[4/7] Enabling behavior monitoring..."
sudo mdatp config behavior-monitoring --value enabled || true

echo
echo "[5/7] Restarting Microsoft Defender service..."
sudo systemctl restart mdatp
sleep 5

echo
echo "[6/7] Verifying Defender connectivity..."
sudo mdatp connectivity test

echo
echo "[7/7] Running quick malware scan..."
sudo mdatp scan quick

echo
echo "=============================="
echo "Final Defender health status"
echo "=============================="
sudo mdatp health --details antivirus
sudo mdatp health --field org_id
sudo mdatp health --field edr_machine_id

echo
echo "=============================="
echo "Installed security-critical packages"
echo "=============================="
dpkg -s \
  screen \
  libpng16-16 \
  libxml2 \
  libxslt1.1 \
  libglib2.0-0 \
  libtasn1-6 \
  libsodium23 \
  gnupg \
  dirmngr \
  rsync \
  zlib1g \
  python3-pyasn1 \
  python3-urllib3 \
  | egrep 'Package:|Version:|Status:'

echo
echo "System hardening complete."
