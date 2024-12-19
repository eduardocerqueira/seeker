#date: 2024-12-19T16:42:47Z
#url: https://api.github.com/gists/c69def14513a3bf5b7b1233ee3830e61
#owner: https://api.github.com/users/schwabix

# identify containername
crictl -r unix:///run/containerd/containerd.sock ps

# get full container id
runc --root /run/containerd/runc/k8s.io/ list | grep xxxxxxx

# enter container as root
runc --root /run/containerd/runc/k8s.io/ exec -t -u 0 xxxxxxx sh