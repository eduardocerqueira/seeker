#date: 2026-02-24T17:52:28Z
#url: https://api.github.com/gists/5168b0e9936e61ed22ad829b03c5170e
#owner: https://api.github.com/users/arm2arm

**reana.yaml**
```yaml
workflow:
  type: serial
  specification:
    steps:
      - name: plot
        container_image: 'gitlab-p4n.aip.de:5005/compute4punch/container-stacks/astro-ml:latest'
        commands:
          - python analyze.py
```

[ Prompt: 61.4 t/s | Generation: 39.7 t/s ]

