#date: 2021-11-01T17:15:27Z
#url: https://api.github.com/gists/959bf3289381f384c8d4d0afe658521a
#owner: https://api.github.com/users/SAHARIPRASAD-2907

# Write a bash script to accept "n" marks (inside a loop), if mark is less than zero ignore it
# and accept again. Further compute the average of "n" marks and display the grade according
# to the condition given below. ‘S grades’ if average > 90 ‘A grade’ if average >= 80 and
# average < 90 ‘B grade’ if average >= 70 and average < 80 ‘C grade’ if average >= 60 and
# average < 70 ‘D grade’ if average >= 55 and average < 60 ‘E grade’ if average >= 50 and
# average < 50 Your script should repeat the same "N" times
echo "input no of students:"
read N
while [ $N -gt 0 ]
do
echo "Enter the number of marks:"
read n
sum=0
i=1
while [ $i -le $n ]
do
echo "enter marks $i "
read m
sum=`expr $sum + $m`
i=`expr $i + 1`
done
avg=`expr $sum / $n`
if [ $avg -gt 90 ]
then
echo "S grades"
elif [ $avg -ge 80 ] && [ $avg -lt 90 ]
then
echo "A grade"
elif [ $avg -ge 70 ] && [ $avg -lt 80 ]
then
echo "B grade"
elif [ $avg -ge 60 ] && [ $avg -lt 70 ]
then
echo "C grade"
elif [ $avg -ge 55 ] && [ $avg -lt 60 ]
then
echo "D grade"
elif [ $avg -ge 50 ] && [ $avg -lt 55 ]
then
echo "E grade"
elif [ $avg -lt 50 ]
then
echo "F grade"
fi
N=`expr $N - 1`
Done