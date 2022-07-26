#date: 2022-07-26T16:45:10Z
#url: https://api.github.com/gists/39bbdce871ce5ba97055e5af32660cb5
#owner: https://api.github.com/users/mastrodim

#!/bin/bash


#Crate random number between two limits.


#RANGE=$((Y-X+1))
#R=$(($(($RANDOM%$RANGE))+X))
#X is the lower limit
#Y is the upper limit
#If B=average blockage interval, then X=B-5 and Y=B+5


while :
do 
  tlow=0.01
  thigh=1000
  B=30
  X=30-5
  Y=30+5
  RANGE=$((Y-X+1))
  R=$(($(($RANDOM%$RANGE))+X))
  sleep $R
  sudo tc class replace dev $(ip route get 10.10.105.2 | grep -oP "(?<=dev )[^ ]+") parent 1: classid 1:3 htb rate "$tlow"mbit
  sleep $R
  sudo tc class replace dev $(ip route get 10.10.105.2 | grep -oP "(?<=dev )[^ ]+") parent 1: classid 1:3 htb rate "$thigh"mbit
done