#date: 2026-02-24T17:52:28Z
#url: https://api.github.com/gists/5168b0e9936e61ed22ad829b03c5170e
#owner: https://api.github.com/users/arm2arm

llama.cpp/build_cuda_amx/bin/llama-cli   -hf $1   --jinja   --reasoning-budget 0   -t 32  --ctx-size 192000   --temp 0.2   --top-p 0.95

**reana.yaml**

```yaml
workflow:
  type: serial
  specification:
    steps:
      - name: getdata
        environment: 'gitlab-p4n.aip.de:5005/compute4punch/container-stacks/astro-ml:latest'
        commands:
          - curl -sO https://s3.data.aip.de:9000/pmviewer2023/particles.cache.npy - name: analysis
        environment: 'gitlab-p4n.aip.de:5005/compute4punch/container-stacks/astro-ml:latest'
        commands:
          - papermill Poster-HUGE-2025.ipynb Poster-HUGE-2025-output.ipynb
          - unlink particles.cache.npy
        kubernetes_memory_limit: '64Gi'
```

[ Prompt: 42.8 t/s | Generation: 31.5 t/s ]
