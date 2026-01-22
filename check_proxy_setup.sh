#!/bin/bash
# Script to identify reverse proxy setup on Azure server
# Run this on your Azure VM: bash check_proxy_setup.sh

echo "=========================================="
echo "Checking Reverse Proxy Configuration"
echo "=========================================="

echo ""
echo "1. Checking for Nginx..."
if command -v nginx &> /dev/null; then
    echo "   ✅ Nginx is installed"
    nginx -v
    echo ""
    echo "   Nginx configuration files:"
    ls -la /etc/nginx/sites-enabled/ 2>/dev/null || echo "   No sites-enabled directory"
    echo ""
    echo "   Current timeout settings:"
    grep -r "proxy_read_timeout\|proxy_connect_timeout\|proxy_send_timeout" /etc/nginx/ 2>/dev/null || echo "   No timeout settings found (using default 60s)"
else
    echo "   ❌ Nginx is NOT installed"
fi

echo ""
echo "2. Checking for Apache..."
if command -v apache2 &> /dev/null || command -v httpd &> /dev/null; then
    echo "   ✅ Apache is installed"
    apache2 -v 2>/dev/null || httpd -v 2>/dev/null
else
    echo "   ❌ Apache is NOT installed"
fi

echo ""
echo "3. Checking for Caddy..."
if command -v caddy &> /dev/null; then
    echo "   ✅ Caddy is installed"
    caddy version
else
    echo "   ❌ Caddy is NOT installed"
fi

echo ""
echo "4. Checking for Traefik (Docker)..."
docker ps 2>/dev/null | grep -i traefik && echo "   ✅ Traefik is running" || echo "   ❌ Traefik not found"

echo ""
echo "5. Checking what's listening on port 443 (HTTPS)..."
sudo ss -tlnp | grep :443 || sudo netstat -tlnp | grep :443 2>/dev/null || echo "   No process found on port 443"

echo ""
echo "6. Checking what's listening on port 80 (HTTP)..."
sudo ss -tlnp | grep :80 || sudo netstat -tlnp | grep :80 2>/dev/null || echo "   No process found on port 80"

echo ""
echo "7. Docker containers status..."
docker ps --format "table {{.Names}}\t{{.Ports}}" 2>/dev/null || echo "   Docker not running or not accessible"

echo ""
echo "8. Azure Network Security Group (for Application Gateway check)..."
echo "   If ports 80/443 are not bound to any process on this VM,"
echo "   you're likely using Azure Application Gateway or Load Balancer."
echo ""
echo "   Check Azure Portal:"
echo "   - Go to: Virtual machines → your-vm → Networking"
echo "   - Look for: Application Gateway or Load Balancer associations"

echo ""
echo "=========================================="
echo "Recommended Actions:"
echo "=========================================="

if nginx -v &> /dev/null; then
    echo ""
    echo "FOR NGINX - Add these lines to /etc/nginx/sites-available/default:"
    echo ""
    echo "    # Inside server { } block, add:"
    echo "    proxy_read_timeout 300s;"
    echo "    proxy_send_timeout 300s;"
    echo "    proxy_connect_timeout 300s;"
    echo ""
    echo "Then run: sudo nginx -t && sudo systemctl reload nginx"
else
    echo ""
    echo "FOR AZURE APPLICATION GATEWAY:"
    echo "1. Go to Azure Portal → Application Gateway"
    echo "2. Select your gateway"
    echo "3. Go to 'Backend settings' → Click on your backend HTTP setting"
    echo "4. Change 'Request timeout' from 30 to 300 seconds"
    echo "5. Save"
fi

echo ""
echo "=========================================="
