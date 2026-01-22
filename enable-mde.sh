#!/usr/bin/env bash
set -e

echo "Disabling passive mode..."
sudo mdatp config passive-mode --value disabled

echo "Enabling real-time protection..."
sudo mdatp config real-time-protection --value enabled

echo "Enabling behavior monitoring..."
sudo mdatp config behavior-monitoring --value enabled

echo "Restarting Microsoft Defender..."
sudo systemctl restart mdatp
sleep 5

echo "Final health check:"
sudo mdatp health --details antivirus
sudo mdatp health --field org_id
sudo mdatp health --field edr_machine_id
