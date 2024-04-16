#date: 2024-04-16T16:48:39Z
#url: https://api.github.com/gists/4fc4e9fb7b97455970b4017e83487652
#owner: https://api.github.com/users/fergalmoran

xfreerdp +clipboard +fonts \
  /sound /mic /smart-sizing \
  /f /floatbar:sticky:off,default:visible,show:fullscreen \
  /scale:180 /scale-desktop:200 \
  /network:auto /cert-ignore \
  /u: "**********":$FM_PASSWORD /v:10.1.1.1 \
  > /dev/null 2>&1 &;
