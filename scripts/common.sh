#!/bin/bash
# https://github.com/techiescamp/kubeadm-scripts/blob/main/scripts/common.sh

set -euxo pipefail

sudo swapoff -a

KUBERNETES_VERSION=v1.31
KUBERNETES_FULL_VERSION=1.31.0-1.1
CRIO_VERSION=v1.30

(crontab -l 2>/dev/null; echo "@reboot /sbin/swapoff -a") | crontab - || true
sudo sed -i '/ swap / s/^/#/' /etc/fstab

sudo apt-get update -y

sudo cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

sudo cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
net.ipv6.conf.all.forwarding        = 1
net.ipv6.conf.default.forwarding    = 1
EOF

sudo sysctl --system

# Install CRI-O and Kubernetes
sudo apt-get update -y
sudo apt-get install -y software-properties-common apt-transport-https ca-certificates curl gpg

curl -fsSL https://pkgs.k8s.io/core:/stable:/$KUBERNETES_VERSION/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/$KUBERNETES_VERSION/deb/ /" | sudo tee /etc/apt/sources.list.d/kubernetes.list

curl -fsSL https://pkgs.k8s.io/addons:/cri-o:/stable:/$CRIO_VERSION/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/cri-o-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/cri-o-apt-keyring.gpg] https://pkgs.k8s.io/addons:/cri-o:/stable:/$CRIO_VERSION/deb/ /" | sudo tee /etc/apt/sources.list.d/cri-o.list

sudo apt-get update -y
sudo apt-get install -y cri-o kubelet="$KUBERNETES_FULL_VERSION" kubectl="$KUBERNETES_FULL_VERSION" kubeadm="$KUBERNETES_FULL_VERSION"
sudo apt-mark hold kubelet kubeadm kubectl

sudo systemctl start crio.service
