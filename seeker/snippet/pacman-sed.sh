#date: 2023-05-26T16:51:40Z
#url: https://api.github.com/gists/9f54272fc092ed2490b15a7c303b01e2
#owner: https://api.github.com/users/ddupas

 #!/bin/sh
 # pacman -Qi shows a lot of info, this one liner selects just the name and description
 # first sed grabs the lines, second adds a blank line, third removes label
 
 pacman -Qi | sed  -n '/Name *: \|Description *: /p' | sed 's/Name/\n&/' | sed  's/Name *: \|Description *: //' | less