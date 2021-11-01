#date: 2021-11-01T17:15:27Z
#url: https://api.github.com/gists/959bf3289381f384c8d4d0afe658521a
#owner: https://api.github.com/users/SAHARIPRASAD-2907

# Write a menu driven bash script (using a case structure) to perform the following. a. Print
# first ‘n’ Triangular numbers b. Check if a number is an Automorphic number c. Check if a
# number is an Abundant number
echo a.triangualar number b.check automorphic number c.check abundant number
read c
case $c in
"a")
echo enter the number of terms:
read n
sum=0
echo
echo the first n triangular numbers are
for ((i=1; $i<=n;i=$i+1))
do
sum=`expr $sum + $i`
echo $sum
done
;;
"b") echo enter the number
read num
(( s=num*num ))
(( N=num ))
while [ $N -gt 0 ]
do
(( lst=N%10 ))
(( st=s%10 ))
if (( lst!= st )); then
f=0
fi
(( N=N/10 ))
(( s=s/10 ))
done
if [ $f -eq 0 ]; then
echo $num is not automorphic number
else
echo $num is automorphic number
fi
;;
"c") echo enter the number
read Num
su=0
for(( y=1;y<Num;y+=1 ))
do
(( f=num%y ))
if (( f==0 )); then
(( su+= y ))
fi
done
if (( su>Num )); then
echo abundant number
else
echo not abundant number
fi
;;
*) echo wrong choice
;;