#date: 2023-11-28T16:48:57Z
#url: https://api.github.com/gists/a12744c26027fcb88e8a96ea9844e257
#owner: https://api.github.com/users/jonprindiville

set -euo pipefail

CLUSTER="${1}"
TASKS="${2}"

aws ecs describe-tasks --cluster "${CLUSTER}" --tasks "${TASKS}" \
| jq '.tasks[]
  | {
      meta: {
        clusterArn: .clusterArn,
        connectivity: .connectivity,
        desiredStatus: .desiredStatus,
        healthStatus: .healthStatus,
        lastStatus: .lastStatus,
        taskDefinitionArn: .taskDefinitionArn
      },
      timing: {
        createdAt: .createdAt|match("[-:0-9T]+").string|strptime("%Y-%m-%dT%H:%M:%S")|mktime,
        connectivityAt: .connectivityAt|match("[-:0-9T]+").string|strptime("%Y-%m-%dT%H:%M:%S")|mktime,
        pullStartedAt: .pullStartedAt|match("[-:0-9T]+").string|strptime("%Y-%m-%dT%H:%M:%S")|mktime,
        pullStoppedAt: .pullStoppedAt|match("[-:0-9T]+").string|strptime("%Y-%m-%dT%H:%M:%S")|mktime,
        startedAt: .startedAt|match("[-:0-9T]+").string|strptime("%Y-%m-%dT%H:%M:%S")|mktime
      }
    } as $result
    | {
      deltas: {
        createdToConnectivity: ($result.timing.connectivityAt - $result.timing.createdAt),
        connectivityToPullStarted: ($result.timing.pullStartedAt - $result.timing.connectivityAt),
        pullStartedToPullStopped: ($result.timing.pullStoppedAt - $result.timing.pullStartedAt),
        pullStoppedToStarted: ($result.timing.startedAt - $result.timing.pullStoppedAt)
      }
    } as $deltaResult
    | {meta: $result.meta, timing: $result.timing, deltas: $deltaResult.deltas}'