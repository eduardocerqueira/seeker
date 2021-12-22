#date: 2021-12-22T16:51:50Z
#url: https://api.github.com/gists/af7ba656d67499645e08959e6c08eb44
#owner: https://api.github.com/users/virusdave

#!/usr/bin/env bash

XL=-2.0;XH=1;YL=-1;YH=1;MI=500;MC=7; for ((y=0;y<$LINES;y++)); do for ((x=0;x<$COLUMNS;x++)); do
 R=$(echo "define g(){scale=10;v=x=((${XH}-(${XL}))*${x}/$COLUMNS+(${XL}));w=y=-((${YH}-(${YL}))*${y}/$LINES+(${YL}));for(i=1;i<${MI};i++){a=v*v-w*w+x;b=2*v*w+y;v=a;w=b;if(v*v+w*w>4){scale=0;return (${MC}-l(i)*(${MC}-1)/l(${MI}))};};return (0)}g()"|bc -l);echo -n -e "\e[4${R}m \e[0m";done;echo;done
