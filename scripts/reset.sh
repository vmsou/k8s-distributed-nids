sudo kubeadm reset

USER_HOME=$(eval echo ~${SUDO_USER})
rm -rf "$USER_HOME"/.kube/
sudo rm -rf /etc/kubernetes/
sudo rm -rf /var/lib/kubelet/
sudo rm -rf /var/lib/etcd
