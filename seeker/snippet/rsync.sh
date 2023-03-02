#date: 2023-03-02T16:39:06Z
#url: https://api.github.com/gists/0e0fa62ad072b58d5a5aed142d200d01
#owner: https://api.github.com/users/dengyt2018

rsync -az -e "ssh -p 22" name@198.168.1.125:/home/test test # rsync via ssh

ssh -o ProxyCommand='nc -X 5 -x 192.168.1.25:1080 %h %p' name@192.168.1.125 -p 22 "dd if=/dev/vda1 | gzip -1 -" | dd of=image.gz
