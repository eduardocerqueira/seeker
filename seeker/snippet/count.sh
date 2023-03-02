#date: 2023-03-02T16:47:36Z
#url: https://api.github.com/gists/90e29057cf076d882d3896bba077053f
#owner: https://api.github.com/users/manjunath111

#!/bin/bash
# Count the total number of nodes
node_count=$(knife node list | wc -l)
# Count the total number of cookbooks
cookbook_count=$(knife cookbook list | wc -l)
# Count the total number of policies
policy_count=$(knife status 'policy_group:*' 'policy_name:*' | grep "Policy revision" | wc -l)
# Print the report
echo "Report generated at $(date)"
echo "----------------------------------------"
echo "Total number of nodes: $node_count"
echo "Total number of cookbooks: $cookbook_count"
echo "Total number of policies: $policy_count"
echo "----------------------------------------"
echo "List of nodes:"
knife node list
echo "----------------------------------------"
echo "List of cookbooks:"
knife cookbook list
echo "----------------------------------------"
echo "List of policies:"
knife status 'policy_group:*' 'policy_name:*'