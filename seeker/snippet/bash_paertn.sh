#date: 2021-11-01T17:15:27Z
#url: https://api.github.com/gists/959bf3289381f384c8d4d0afe658521a
#owner: https://api.github.com/users/SAHARIPRASAD-2907

echo enter the number
read n
for(( i=1;i<=n;i+=1 ))
do
  for(( j=n-i+1;j>0;j-=1 ))
do
printf " "
done
for(( k=1;k<=i;k+=1 ))
do
printf "%d" $k
done
for(( x=i-1;x>0;x-=1 ))
do
printf "%d" $x
done
echo
done