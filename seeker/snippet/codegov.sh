#date: 2024-05-13T16:52:37Z
#url: https://api.github.com/gists/7e11d24f21536f4fd2e8687f30b4a0a4
#owner: https://api.github.com/users/jwiegley

#!/bin/bash

URL='https://ic-api.internetcomputer.org/api/v3/proposals?include_action=ExecuteNnsFunction&offset=0&include_topic=TOPIC_IC_OS_VERSION_ELECTION&limit=50&include_status=OPEN&include_reward_status=ACCEPT_VOTES'

if [[ ! -f proposals_seen ]]; then
    touch proposals_seen
fi

echo "Checking for new proposals at $(date)"

for proposal in $(curl -sX GET "$URL" -H 'accept: application/json' | jq -r '.data[].proposal_id')
do
    if ! grep -q "^$proposal\$" proposals_seen
    then
        echo $proposal >> proposals_seen
        cat > message.txt <<EOF
From: jwiegley@gmail.com
To: codegov@googlegroups.com
Subject: Replica Upgrade Proposal: $proposal

There is a new replica upgrade proposal available:

  https://dashboard.internetcomputer.org/proposal/$proposal
EOF
        cat message.txt | msmtp -C ./msmtp.config --read-envelope-from --read-recipients
    fi
done

URL='https://ic-api.internetcomputer.org/api/v3/proposals?offset=0&include_topic=TOPIC_GOVERNANCE&include_topic=TOPIC_SNS_AND_COMMUNITY_FUND&limit=50&include_status=OPEN&include_reward_status=ACCEPT_VOTES'

if [[ ! -f governance_seen ]]; then
    touch governance_seen
fi

echo "Checking for new governance proposals at $(date)"

for proposal in $(curl -sX GET "$URL" -H 'accept: application/json' | jq -r '.data[].proposal_id')
do
    if ! grep -q "^$proposal\$" governance_seen
    then
        echo $proposal >> governance_seen
        cat > message.txt <<EOF
From: jwiegley@gmail.com
To: synapsevote@googlegroups.com
Subject: Governance or SNS/SF Proposal: $proposal

There is a new governance or SNS/SF proposal available:

  https://dashboard.internetcomputer.org/proposal/$proposal
EOF
        cat message.txt | msmtp -C ./msmtp.config --read-envelope-from --read-recipients
    fi
done

