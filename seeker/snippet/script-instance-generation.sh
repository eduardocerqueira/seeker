#date: 2022-02-10T16:53:51Z
#url: https://api.github.com/gists/43e65370a208ed3b17255643ae4efa40
#owner: https://api.github.com/users/cexposit

#!/bin/bash

# Directory with the instances to solve
INPUT_DIRECTORY=/home_nfs/cei/QAP/instances/

for inputFile in $INPUT_DIRECTORY*
do
	arr=($(echo $inputFile | tr "/" "\n"))
	lenghtArray=${#arr[*]}
	instance=${arr[lenghtArray-1]}
	individuals=200
	for elite in {5..15..5}
	do
		for mutants in {5..15..5}
		do 
			for generations in {50..200..50}
			do 
				for crossover in {2..8..2}
				do
					java -cp .:./MyQAP.jar Main $inputFile $individuals $elite $mutants $generations $crossover 10 > instance${individuals}_${mutants}.json
				done
			done
		done
	done
done