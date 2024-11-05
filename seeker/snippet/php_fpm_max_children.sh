#date: 2024-11-05T17:03:17Z
#url: https://api.github.com/gists/ccfe441a1be9297c4c7a5ba33aed5102
#owner: https://api.github.com/users/FerraBraiZ

#!/bin/bash

# Get memory usage per child process
memory_usage_per_child=$(ps --no-headers -o "rss,cmd" -C php-fpm | awk '{ sum+=$1 } END { printf ("%d\n", sum/NR/1024) }');
echo
echo "Average memory usage per child: ${memory_usage_per_child}MB"

# Get total available memory
total_available_memory=$(free -m | awk 'NR==2{print $2}');
total_available_memory_gb=$(echo "scale=2; $total_available_memory / 1024" | bc)
echo
echo "Total available memory: ${total_available_memory_gb}GB"

# Calculate max children
max_children=$((total_available_memory / memory_usage_per_child))

# Multiply max_children by 0.8
max_children_multiplier=0.8
max_children_limit=$(echo "scale=0; $max_children * $max_children_multiplier" | bc)


# Print the result
echo "Sugestions: "
echo
echo " pm.max_children=${max_children} (100% of the whole memory)"
echo " pm.max_children=${max_children_limit} (80% of the available memory)"