#date: 2023-08-22T17:02:10Z
#url: https://api.github.com/gists/1755bc9f1b628886605186407064087c
#owner: https://api.github.com/users/alexeldeib

# install nvidia device plugin (without env var)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin-compat-with-cpumanager.yml

# ssh OR nsenter node using node-shell + privileged pod
# tried both to eliminate any container mount issues.
# same behavior
# https://github.com/kvaps/kubectl-node-shell
kubectl node-shell aks-nca100-36400834-vmss000000


# in a separate shell:
# write out the test manifest
tee nvidia-smi-loop.yaml > /dev/null <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: cuda-nvidia-smi-loop
spec:
  restartPolicy: OnFailure
  containers:
  - name: cuda
    image: "nvcr.io/nvidia/cuda:12.0.0-base-ubuntu20.04"
    command: ["/bin/sh", "-c"]
    args: ["while true; do nvidia-smi -L; sleep 5; done"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

kubectl apply -f nvidia-smi-loop.yaml  

# back in node-shell, trigger the issue
systemctl daemon-reload

kubectl logs -f cuda-nvidia-smi-loop
# GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-0451fe54-e0a1-36c5-eeb5-19025f49663e)
# GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-0451fe54-e0a1-36c5-eeb5-19025f49663e)
# GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-0451fe54-e0a1-36c5-eeb5-19025f49663e)
# GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-0451fe54-e0a1-36c5-eeb5-19025f49663e)
# Failed to initialize NVML: Unknown Error
# Failed to initialize NVML: Unknown Error

