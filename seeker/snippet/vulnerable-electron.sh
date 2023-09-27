#date: 2023-09-27T17:00:11Z
#url: https://api.github.com/gists/3da7c3720b0d9f3ee7dc9a95f623578d
#owner: https://api.github.com/users/april

for filename in /Applications/*.app/Contents/Frameworks/Electron\ Framework.framework/Electron\ Framework
do
  echo -n "$filename: "
  strings $filename | grep "Chrome/" | grep -i Electron | grep -v '%s' | sort -u
done
