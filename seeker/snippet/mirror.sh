#date: 2025-04-11T16:53:43Z
#url: https://api.github.com/gists/2e8a6907ad1f8ce136d8eabef6303df6
#owner: https://api.github.com/users/erdii

#!/bin/bash
set -euxo pipefail

skopeo copy --all \
  docker://quay.io/package-operator/package-operator-package:v1.18.2 \
  docker://quay.io/erdii-test/pko-mirror/package-operator-package:v1.18.2

skopeo copy --all \
  docker://quay.io/package-operator/package-operator-manager:v1.18.2 \
  docker://quay.io/erdii-test/pko-mirror/package-operator-manager:v1.18.2

skopeo copy --all \
  docker://quay.io/package-operator/remote-phase-manager:v1.18.2 \
  docker://quay.io/erdii-test/pko-mirror/remote-phase-manager:v1.18.2

curl -LO https://github.com/package-operator/package-operator/releases/download/v1.18.2/self-bootstrap-job.yaml

yq -i 'with(
      select(.kind == "Job").spec.template.spec.containers[]
    | select(.name == "package-operator").env[]
    | select(.name == "PKO_IMAGE_PREFIX_OVERRIDES")
    ; .value = "quay.io/package-operator/=quay.io/erdii-test/pko-mirror/"
  )' self-bootstrap-job.yaml

yq -i 'with(
      select(.kind == "Job").spec.template.spec.containers[]
    | select(.name == "package-operator")
    ; (
      .image |= sub("quay.io/package-operator", "quay.io/erdii-test/pko-mirror"),
      .args[0] |= sub("quay.io/package-operator", "quay.io/erdii-test/pko-mirror")
    )
  )' self-bootstrap-job.yaml
