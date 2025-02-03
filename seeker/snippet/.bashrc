#date: 2025-02-03T16:39:17Z
#url: https://api.github.com/gists/a64714b340e9d5a0c3943163b8272fbe
#owner: https://api.github.com/users/chicagobuss

# SSH Agent
if [[ ! $(ps -ef | grep "[s]sh-agent") ]]; then
  echo "Starting SSH Agent"
  eval $(ssh-agent -s)
else
  ssh_agent_pid=$(pidof ssh-agent)
  socket=$(find /tmp/ssh* | grep agent)
  short_sock=$(find /tmp/ssh* | grep agent | cut -d '.' -f 2)
  if [[ "${short_sock::-2}" == "${ssh_agent_pid::-2}" ]]; then
    echo "Found ssh-agent running with pid ${ssh_agent_pid} and matching"
    echo "socket ${socket}, setting vars SSH_AUTH_SOCK and SSH_AGENT_PID"
    export SSH_AUTH_SOCK=${socket}
    export SSH_AGENT_PID=${ssh_agent_pid}
  else
    echo "Found ssh-agent pid but no matching socket. Here's what I found:"
    echo " agent pid(s): ${ssh_agent_pid}"
    echo " socket: ${socket}"
  fi
fi
####