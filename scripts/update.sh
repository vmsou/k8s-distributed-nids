#!/bin/bash

# Check if a hostname was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <hostname>"
    exit 1
fi

sudo hostnamectl set-hostname "$1"

# Update /etc/hosts
cp /etc/hosts /etc/hosts.bak

sed -i "/127.0.1.1/d" /etc/hosts
echo -e "127.0.1.1       $1" >> /etc/hosts

echo "Hostname set to $1 and /etc/hosts updated."

# Update node ip
sudo apt-get install -y jq
local_ip="$(ip --json addr show eth0 | jq -r '.[0].addr_info[] | select(.family == "inet") | .local')"
sudo cat > /etc/default/kubelet << EOF
KUBELET_EXTRA_ARGS=--node-ip=$local_ip
EOF
