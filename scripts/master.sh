#!/bin/bash
# https://github.com/techiescamp/kubeadm-scripts/blob/main/scripts/master.sh

set -euxo pipefail

# Check if a IP_TYPE was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <public|private>"
    exit 1
fi

IP_TYPE="$1"
NODENAME=$(hostname -s)
POD_CIDR="192.168.0.0/16"

sudo kubeadm config images pull

if [[ "$IP_TYPE" == "private" ]]; then
    MASTER_PRIVATE_IP=$(ip addr show eth0 | awk '/inet / {print $2}' | cut -d/ -f1)
    sudo kubeadm init --apiserver-advertise-address="$MASTER_PRIVATE_IP" --apiserver-cert-extra-sans="$MASTER_PRIVATE_IP" --pod-network-cidr="$POD_CIDR" --node-name "$NODENAME" --ignore-preflight-errors Swap

elif [[ "$IP_TYPE" == "public" ]]; then
    MASTER_PUBLIC_IP=$(curl ipinfo.io/ip && echo "")
    sudo kubeadm init --control-plane-endpoint="$MASTER_PUBLIC_IP" --apiserver-cert-extra-sans="$MASTER_PUBLIC_IP" --pod-network-cidr="$POD_CIDR" --node-name "$NODENAME" --ignore-preflight-errors Swap

else
    echo "Error: IP_TYPE has an invalid value: $IP_TYPE"
    exit 1
fi

USER_HOME=$(eval echo ~${SUDO_USER})
mkdir -p "$USER_HOME/.kube"
sudo cp -i /etc/kubernetes/admin.conf "$USER_HOME/.kube/config"
sudo chown "$(id -u ${SUDO_USER})":"$(id -g ${SUDO_USER})" "$USER_HOME/.kube/config"

# kubectl apply -f https://raw.githubusercontent.com/antrea-io/antrea/main/build/yamls/antrea.yml
# kubectl apply -f https://docs.projectcalico.org/manifests/calico.yaml
# kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

