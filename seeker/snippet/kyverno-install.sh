#date: 2021-09-15T16:58:19Z
#url: https://api.github.com/gists/739782df80d684d96370e7cc3c0f6a0e
#owner: https://api.github.com/users/arun1801

# Add Kyverno-crds
helm install kyverno-crds kyverno/kyverno-crds --namespace kyverno --create-namespace

# Install Kyverno
helm install kyverno kyverno/kyverno --namespace kyverno