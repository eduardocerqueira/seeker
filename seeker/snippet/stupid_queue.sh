#date: 2021-11-09T17:14:31Z
#url: https://api.github.com/gists/5fc9f623b3e6ce8b4df4c4c66385cc13
#owner: https://api.github.com/users/jlabounty

# While running on a cluster, I needed a simple queue system in order to prevent the hard disk from crashing 
# due to 100+ file transfers requested all at once. This snippet alleviated the issue without requiring an 
# external program to be running to keep track of things, only in bash

# Josh LaBounty 
# 2020

# setup work....

queue_dir="/path/to/accessable_directory/"

echo "Copying files..."
while :; do
	nfiles=$(ls -l ${queue_dir} | wc -l)
	echo "$nfiles in directory"

	if [ "$nfiles" -le "5" ]; then # adjust this number based on your file system. 
		echo "     -> Processing!"
		touch ${queue_dir}${datestring}

		# scp ${input_files} ${working_directory}

		rm -f ${queue_dir}${datestring}
		break
	fi
	echo "     queue is full -> waiting..."
	sleep 10
done
echo "Copied!"

# do work on cluster node...