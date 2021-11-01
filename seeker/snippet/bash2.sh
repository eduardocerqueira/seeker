#date: 2021-11-01T17:15:27Z
#url: https://api.github.com/gists/959bf3289381f384c8d4d0afe658521a
#owner: https://api.github.com/users/SAHARIPRASAD-2907

# Write a bash script to display all files in the /home/YourLoginName subdirectory as well
# as display the type of all files. If the file is an ordinary file print its permission and change the
# permissions to r - - r - - r - -

code:
for file in `ls`
do
if [ -d $file ]
then
echo $file is a directory
elif [ -c $file ]
then
echo $file is a character device file
elif [ -b $file ]
then
echo $file is a block device file
elif [ -S $file ]
then
echo $file is a domain socket
elif [ -p $file ]
then
echo $file is a named pipe

elif [ -L $file ]
then
echo $file is a symbolic link
elif [ -f $file ]
then
echo $file is a regular file
chmod -wx $file
else
echo $file
fi
done