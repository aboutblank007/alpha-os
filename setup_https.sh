#!/bin/bash

SERVER="alphaos"
DOMAIN="alphaos.lootool.cn"
EMAIL="admin@lootool.cn" # Replace with your email if needed
UPSTREAM_PORT="3001"

echo "🚀 Setting up HTTPS for $DOMAIN on $SERVER..."

# 1. Create Nginx Config File locally
echo "📄 Generating Nginx configuration..."
cat > alphaos_nginx.conf <<EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:$UPSTREAM_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# 2. Upload and Execute Setup on Server
echo "📤 Uploading configuration and executing setup script..."

# Create a remote setup script
cat > remote_setup.sh <<EOF
#!/bin/bash
set -e

DOMAIN="$DOMAIN"
EMAIL="$EMAIL"

# Install Nginx and Certbot if not present
if ! command -v nginx &> /dev/null; then
    echo "📦 Installing Nginx and Certbot..."
    sudo apt-get update
    sudo apt-get install -y nginx certbot python3-certbot-nginx
fi

# Copy Nginx config
echo "⚙️ Configuring Nginx..."
sudo mv ~/alphaos_nginx.conf /etc/nginx/sites-available/\$DOMAIN
sudo ln -sf /etc/nginx/sites-available/\$DOMAIN /etc/nginx/sites-enabled/

# Remove default if it exists (optional, to avoid conflicts)
# sudo rm -f /etc/nginx/sites-enabled/default

# Test and Reload Nginx
sudo nginx -t
sudo systemctl reload nginx

# Obtain SSL Certificate
echo "🔒 Obtaining SSL Certificate..."
sudo certbot --nginx -d \$DOMAIN --non-interactive --agree-tos -m \$EMAIL --redirect

echo "✅ HTTPS Setup Complete! Visit https://\$DOMAIN"
EOF

# Transfer files
scp alphaos_nginx.conf $SERVER:~/alphaos_nginx.conf
scp remote_setup.sh $SERVER:~/remote_setup.sh

# Execute
ssh $SERVER "chmod +x ~/remote_setup.sh && ~/remote_setup.sh"

# Cleanup local files
rm alphaos_nginx.conf remote_setup.sh

echo "🎉 Done!"

