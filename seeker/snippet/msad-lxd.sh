#date: 2023-07-05T16:39:03Z
#url: https://api.github.com/gists/e0edd9bf6c8d52ec35ce434b93289461
#owner: https://api.github.com/users/ThinGuy

#!/bin/bash

export DNS_DOMAIN="atx.orangebox.me"
export MSAD_DOMAIN="ORANGEBOX"
export DC_HOSTNAME="msadc"
export SSH_IMPORT_ID="lp:craig-bender, gh:thinguy"
export UBUNTU_REPO="us.archive.ubuntu.com"
export UBUNTU_RELEASE="lunar"
export MSADMIN_PW="Ubuntu1+"
export KRBTGT_PW="Ubuntu1+"
export TZ="America/Los_Angeles"
export LOCALE="en_US.UTF-8"
export LXD_CHILD_NIC="eth0"
export LXD_PARENT_NIC="br0"
export LXD_STORAGE_POOL="default"
export LXD_PROFILE_NAME="msad-demo"
export DNS1="172.27.44.1"
export DNS2="172.27.46.1"
export DNS_FWDER="1.1.1.1"
export DNS_SEARCH="${DNS_DOMAIN}, orangebox.me"

lxc 2>/dev/null profile create ${LXD_PROFILE_NAME}
cat <<LXDPROF |sed -r 's/[ \t]+$//g'|lxc profile edit ${LXD_PROFILE_NAME}
config:
  boot.autostart: "true"
  security.nesting: "true"
  security.privileged: "false"
  user.network-config: |
    version: 2
    ethernets:
      ${LXD_CHILD_NIC}:
        dhcp4: false
        dhcp6: false
        accept-ra: false
        optional: false
        mtu: 1472
    bridges:
      br0:
        interfaces: [${LXD_CHILD_NIC}]
        mtu: 1472
        dhcp4: true
        dhcp4-overrides:
          use-dns: false
          use-hostname: false
          use-domains: false
          route-metric: 0
        dhcp6: true
        dhcp6-overrides:
          use-dns: false
          use-hostname: false
          use-domains: false
          route-metric: 0
        accept-ra: false
        optional: false
        nameservers:
          addresses:
           - ${DNS1}
           - ${DNS2}
           - ${DNS_FWDER}
          search: [${DNS_SEARCH}]
        parameters:
          priority: 0
          stp: false
  user.user-data: |
    #cloud-config
    merge_how:
      - name: list
        settings: [append]
      - name: dict
        settings: [no_replace, recurse_list]
    final_message: 'MSAD Controller Completed in \$UPTIME'
    manage_etc_hosts: true
    preserve_hostname: true
    prefer_fqdn_over_hostname: true
    manage_resolv_conf: true
    hostname: ${DC_HOSTNAME}
    fqdn: ${DC_HOSTNAME}.${DNS_DOMAIN}
    write_files:
      - encoding: b64
        content: bWFjaGluZSBwcml2YXRlLXBwYS5sYXVuY2hwYWRjb250ZW50Lm5ldC9jcmJzL3Byb3Bvc2VkL3VidW50dSBsb2dpbiBjcmFpZy1iZW5kZXIgcGFzc3dvcmQgR0wwMU1iWjgwOUJuMUJiaDRHYzQK
        owner: 'root:root'
        path: /etc/apt/auth.conf.d/99canonical-rbac
        permissions: '0600'
      - encoding: b64
        content: ZG46IG91PVVzZXJzLGRjPWRlbW8sZGM9b3JhbmdlYm94LGRjPW1lCm9iamVjdENsYXNzOiBvcmdhbml6YXRpb25hbFVuaXQKb3U6IFVzZXJzCgpkbjogb3U9QWRtaW5pc3RyYXRvcnMsZGM9ZGVtbyxkYz1vcmFuZ2Vib3gsZGM9bWUKb2JqZWN0Q2xhc3M6IG9yZ2FuaXphdGlvbmFsVW5pdApvdTogQWRtaW5pc3RyYXRvcnMKCmRuOiBvdT1TZXJ2aWNlQWNjb3VudHMsZGM9ZGVtbyxkYz1vcmFuZ2Vib3gsZGM9bWUKb2JqZWN0Q2xhc3M6IG9yZ2FuaXphdGlvbmFsVW5pdApvdTogU2VydmljZUFjY291bnRzCgpkbjogb3U9R3JvdXBzLGRjPWRlbW8sZGM9b3JhbmdlYm94LGRjPW1lCm9iamVjdENsYXNzOiBvcmdhbml6YXRpb25hbFVuaXQKb3U6IEdyb3VwcwoKZG46IHVpZD1ycGMtZGVtby1iaW5kLG91PVNlcnZpY2VBY2NvdW50cyxkYz1kZW1vLGRjPW9yYW5nZWJveCxkYz1tZQpvYmplY3RDbGFzczogaW5ldE9yZ1BlcnNvbgp1aWQ6IHJwYy1kZW1vLWJpbmQKc246IHJwYy1kZW1vLWJpbmQKZ2l2ZW5OYW1lOiBycGMtZGVtby1iaW5kCmNuOiBycGMtZGVtby1iaW5kCmRpc3BsYXlOYW1lOiBycGMtZGVtby1iaW5kCnVzZXJQYXNzd29yZDogcnBjLWRlbW8tYmluZAoKZG46IHVpZD1kZW1vLXN0YWNrX2RvbWFpbl9hZG1pbixvdT1Vc2VycyxkYz1kZW1vLGRjPW9yYW5nZWJveCxkYz1tZQpvYmplY3RDbGFzczogaW5ldE9yZ1BlcnNvbgp1aWQ6ZG9tYWluX2FkbWluCnNuOmRvbWFpbl9hZG1pbgpnaXZlbk5hbWU6ZG9tYWluX2FkbWluCmNuOmRvbWFpbl9hZG1pbgpkaXNwbGF5TmFtZTpkb21haW5fYWRtaW4KdXNlclBhc3N3b3JkOmRvbWFpbl9hZG1pbgoKZG46IHVpZD1kZW1vLWFkbWluLG91PVVzZXJzLGRjPWRlbW8sZGM9b3JhbmdlYm94LGRjPW1lCm9iamVjdENsYXNzOiBpbmV0T3JnUGVyc29uCnVpZDogZGVtby1hZG1pbgpzbjogZGVtby1hZG1pbgpjbjogZGVtby1hZG1pbgpkaXNwbGF5TmFtZTogZGVtby1hZG1pbgp1c2VyUGFzc3dvcmQ6IGRlbW8tYWRtaW4KCmRuOiB1aWQ9ZGVtby1rZXlzdG9uZSxvdT1Vc2VycyxkYz1kZW1vLGRjPW9yYW5nZWJveCxkYz1tZQpvYmplY3RDbGFzczogaW5ldE9yZ1BlcnNvbgp1aWQ6IGRlbW8ta2V5c3RvbmUKc246IGRlbW8ta2V5c3RvbmUKY246IGRlbW8ta2V5c3RvbmUKZGlzcGxheU5hbWU6IGRlbW8ta2V5c3RvbmUKdXNlclBhc3N3b3JkOiBkZW1vLWtleXN0b25lCgpkbjogdWlkPWRlbW8tc3dpZnRfZGlzcGVyc2lvbixvdT1Vc2VycyxkYz1kZW1vLGRjPW9yYW5nZWJveCxkYz1tZQpvYmplY3RDbGFzczogaW5ldE9yZ1BlcnNvbgp1aWQ6IGRlbW8tc3dpZnRfZGlzcGVyc2lvbgpzbjogZGVtby1zd2lmdF9kaXNwZXJzaW9uCmNuOiBkZW1vLXN3aWZ0X2Rpc3BlcnNpb24KZGlzcGxheU5hbWU6IGRlbW8tc3dpZnRfZGlzcGVyc2lvbgp1c2VyUGFzc3dvcmQ6IGRlbW8tc3dpZnRfZGlzcGVyc2lvbgoKZG46IHVpZD1kZW1vLXN3aWZ0LG91PVVzZXJzLGRjPWRlbW8sZGM9b3JhbmdlYm94LGRjPW1lCm9iamVjdENsYXNzOiBpbmV0T3JnUGVyc29uCnVpZDogZGVtby1zd2lmdApzbjogZGVtby1zd2lmdApjbjogZGVtby1zd2lmdApkaXNwbGF5TmFtZTogZGVtby1zd2lmdAp1c2VyUGFzc3dvcmQ6IGRlbW8tc3dpZnQKCmRuOiB1aWQ9ZGVtby1jaW5kZXIsb3U9VXNlcnMsZGM9ZGVtbyxkYz1vcmFuZ2Vib3gsZGM9bWUKb2JqZWN0Q2xhc3M6IGluZXRPcmdQZXJzb24KdWlkOiBkZW1vLWNpbmRlcgpzbjogZGVtby1jaW5kZXIKY246IGRlbW8tY2luZGVyCmRpc3BsYXlOYW1lOiBkZW1vLWNpbmRlcgp1c2VyUGFzc3dvcmQ6IGRlbW8tY2luZGVyCgpkbjogdWlkPWRlbW8tZ2xhbmNlLG91PVVzZXJzLGRjPWRlbW8sZGM9b3JhbmdlYm94LGRjPW1lCm9iamVjdENsYXNzOiBpbmV0T3JnUGVyc29uCnVpZDogZGVtby1nbGFuY2UKc246IGRlbW8tZ2xhbmNlCmNuOiBkZW1vLWdsYW5jZQpkaXNwbGF5TmFtZTogZGVtby1nbGFuY2UKdXNlclBhc3N3b3JkOiBkZW1vLWdsYW5jZQoKZG46IHVpZD1kZW1vLWhlYXQsb3U9VXNlcnMsZGM9ZGVtbyxkYz1vcmFuZ2Vib3gsZGM9bWUKb2JqZWN0Q2xhc3M6IGluZXRPcmdQZXJzb24KdWlkOiBkZW1vLWhlYXQKc246IGRlbW8taGVhdApjbjogZGVtby1oZWF0CmRpc3BsYXlOYW1lOiBkZW1vLWhlYXQKdXNlclBhc3N3b3JkOiBkZW1vLWhlYXQKCmRuOiB1aWQ9ZGVtby1uZXV0cm9uLG91PVVzZXJzLGRjPWRlbW8sZGM9b3JhbmdlYm94LGRjPW1lCm9iamVjdENsYXNzOiBpbmV0T3JnUGVyc29uCnVpZDogZGVtby1uZXV0cm9uCnNuOiBkZW1vLW5ldXRyb24KY246IGRlbW8tbmV1dHJvbgpkaXNwbGF5TmFtZTogZGVtby1uZXV0cm9uCnVzZXJQYXNzd29yZDogZGVtby1uZXV0cm9uCgpkbjogdWlkPWRlbW8tbm92YSxvdT1Vc2VycyxkYz1kZW1vLGRjPW9yYW5nZWJveCxkYz1tZQpvYmplY3RDbGFzczogaW5ldE9yZ1BlcnNvbgp1aWQ6IGRlbW8tbm92YQpzbjogZGVtby1ub3ZhCmNuOiBkZW1vLW5vdmEKZGlzcGxheU5hbWU6IGRlbW8tbm92YQp1c2VyUGFzc3dvcmQ6IGRlbW8tbm92YQoKZG46IHVpZD1kZW1vLWNlaWxvbWV0ZXIsb3U9VXNlcnMsZGM9ZGVtbyxkYz1vcmFuZ2Vib3gsZGM9bWUKb2JqZWN0Q2xhc3M6IGluZXRPcmdQZXJzb24KdWlkOiBkZW1vLWNlaWxvbWV0ZXIKc246IGRlbW8tY2VpbG9tZXRlcgpnaXZlbk5hbWU6IGRlbW8tY2VpbG9tZXRlcgpjbjogZGVtby1jZWlsb21ldGVyCmRpc3BsYXlOYW1lOiBkZW1vLWNlaWxvbWV0ZXIKdXNlclBhc3N3b3JkOiBkZW1vLWNlaWxvbWV0ZXIKCmRuOiB1aWQ9Y2xvdWQtdXNlci0xLG91PVVzZXJzLGRjPWRlbW8sZGM9b3JhbmdlYm94LGRjPW1lCm9iamVjdENsYXNzOiBpbmV0T3JnUGVyc29uCnVpZDogY2xvdWQtdXNlci0xCnNuOiBjbG91ZC11c2VyLTEKZ2l2ZW5OYW1lOiBjbG91ZC11c2VyLTEKY246IGNsb3VkLXVzZXItMQpkaXNwbGF5TmFtZTogY2xvdWQtdXNlci0xCnVzZXJQYXNzd29yZDogY2xvdWQtdXNlci0xCgpkbjogdWlkPWNsb3VkLXVzZXItMixvdT1Vc2VycyxkYz1kZW1vLGRjPW9yYW5nZWJveCxkYz1tZQpvYmplY3RDbGFzczogaW5ldE9yZ1BlcnNvbgp1aWQ6IGNsb3VkLXVzZXItMgpzbjogY2xvdWQtdXNlci0yCmdpdmVuTmFtZTogY2xvdWQtdXNlci0yCmNuOiBjbG91ZC11c2VyLTIKZGlzcGxheU5hbWU6IGNsb3VkLXVzZXItMgp1c2VyUGFzc3dvcmQ6IGNsb3VkLXVzZXItMgoK
        owner: 'root:root'
        path: /root/ldap.diff
        permissions: '0600'
      - encoding: b64
        content: IyEvYmluL2Jhc2gKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJwYWRtw6kuYW1pZGFsYSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJQYWRtw6kiIC0tc3VybmFtZT0iQW1pZGFsYSIgLS1pbml0aWFscz0icGEiIC0tbWFpbC1hZGRyZXNzPSJwYWRtw6kuYW1pZGFsYUBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IlBhZG3DqSBBbWlkYWxhIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImNhc3NpYW4uYW5kb3IiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iQ2Fzc2lhbiIgLS1zdXJuYW1lPSJBbmRvciIgLS1pbml0aWFscz0iY2EiIC0tbWFpbC1hZGRyZXNzPSJjYXNzaWFuLmFuZG9yQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQ2Fzc2lhbiBBbmRvciIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJ3ZWRnZS5hbnRpbGxlcyIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJXZWRnZSIgLS1zdXJuYW1lPSJBbnRpbGxlcyIgLS1pbml0aWFscz0id2EiIC0tbWFpbC1hZGRyZXNzPSJ3ZWRnZS5hbnRpbGxlc0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IldlZGdlIEFudGlsbGVzIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgInRvYmlhcy5iZWNrZXR0IiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IlRvYmlhcyIgLS1zdXJuYW1lPSJCZWNrZXR0IiAtLWluaXRpYWxzPSJ0YiIgLS1tYWlsLWFkZHJlc3M9InRvYmlhcy5iZWNrZXR0QHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iVG9iaWFzIEJlY2tldHQiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiamFyLmJpbmtzIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkphciIgLS1zdXJuYW1lPSJCaW5rcyIgLS1pbml0aWFscz0iampiIiAtLW1haWwtYWRkcmVzcz0iamFyLmJpbmtzQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iSmFyIEphciBCaW5rcyIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJsYW5kby5jYWxyaXNzaWFuIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkxhbmRvIiAtLXN1cm5hbWU9IkNhbHJpc3NpYW4iIC0taW5pdGlhbHM9ImxjIiAtLW1haWwtYWRkcmVzcz0ibGFuZG8uY2Fscmlzc2lhbkBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkxhbmRvIENhbHJpc3NpYW4iCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiY2hld3kiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iQ2hld2JhY2NhIiAtLWluaXRpYWxzPSJjIiAtLW1haWwtYWRkcmVzcz0iY2hld3lAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJDaGV3eSIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJwb2UuZGFtZXJvbiIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJQb2UiIC0tc3VybmFtZT0iRGFtZXJvbiIgLS1pbml0aWFscz0icGQiIC0tbWFpbC1hZGRyZXNzPSJwb2UuZGFtZXJvbkBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IlBvZSBEYW1lcm9uIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImRhcnRoLnR5cmFudXMiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iQ291bnQiIC0tc3VybmFtZT0iRG9va3UiIC0taW5pdGlhbHM9ImNkIiAtLW1haWwtYWRkcmVzcz0iZGFydGgudHlyYW51c0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkRhcnRoIFR5cmFudXMiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAianluLmVyc28iIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iSnluIiAtLXN1cm5hbWU9IkVyc28iIC0taW5pdGlhbHM9ImplIiAtLW1haWwtYWRkcmVzcz0ianluLmVyc29Ac3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJKeW4gRXJzbyIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJib2JhLmZldHQiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iQm9iYSIgLS1zdXJuYW1lPSJGZXR0IiAtLWluaXRpYWxzPSJiZiIgLS1tYWlsLWFkZHJlc3M9ImJvYmEuZmV0dEBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkJvYmEgRmV0dCIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJqYW5nby5mZXR0IiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkphbmdvIiAtLXN1cm5hbWU9IkZldHQiIC0taW5pdGlhbHM9ImpmIiAtLW1haWwtYWRkcmVzcz0iamFuZ28uZmV0dEBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkphbmdvIEZldHQiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiZm4tMjE4NyIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJGaW5uIiAtLWluaXRpYWxzPSJmbiIgLS1tYWlsLWFkZHJlc3M9ImZuLTIxODdAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJGTi0yMTg3IgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImJpYi5mb3J0dW5hIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkJpYiIgLS1zdXJuYW1lPSJGb3J0dW5hIiAtLWluaXRpYWxzPSJiZiIgLS1tYWlsLWFkZHJlc3M9ImJpYi5mb3J0dW5hQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQmliIEZvcnR1bmEiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAic2F3LmdlcnJlcmEiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iU2F3IiAtLXN1cm5hbWU9IkdlcnJlcmEiIC0taW5pdGlhbHM9InNnIiAtLW1haWwtYWRkcmVzcz0ic2F3LmdlcnJlcmFAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJTYXcgR2VycmVyYSIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJncmVlZG8iIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iR3JlZWRvIiAtLWluaXRpYWxzPSJnIiAtLW1haWwtYWRkcmVzcz0iZ3JlZWRvLkBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkdyZWVkbyIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJqYWJiYS5odXR0IiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkphYmJhIiAtLXN1cm5hbWU9Ikh1dHQiIC0taW5pdGlhbHM9Imp0aCIgLS1tYWlsLWFkZHJlc3M9ImphYmJhLmh1dHRAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJKYWJiYSB0aGUgSHV0dCIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJnZW5lcmFsLmh1eCIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJHZW5lcmFsIiAtLXN1cm5hbWU9Ikh1eCIgLS1pbml0aWFscz0iZ2giIC0tbWFpbC1hZGRyZXNzPSJnZW5lcmFsLmh1eEBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkdlbmVyYWwgSHV4IgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgInF1aS1nb24uamlubiIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJRdWktR29uIiAtLXN1cm5hbWU9Ikppbm4iIC0taW5pdGlhbHM9InFqIiAtLW1haWwtYWRkcmVzcz0icXVpLWdvbi5qaW5uQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iUXVpLUdvbiBKaW5uIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgIm1hei5rYW5hdGEiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iTWF6IiAtLXN1cm5hbWU9IkthbmF0YSIgLS1pbml0aWFscz0ibWsiIC0tbWFpbC1hZGRyZXNzPSJtYXoua2FuYXRhQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iTWF6IEthbmF0YSIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJvYmktd2FuLmtlbm9iaSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJPYmktV2FuIiAtLXN1cm5hbWU9Iktlbm9iaSIgLS1pbml0aWFscz0ib2siIC0tbWFpbC1hZGRyZXNzPSJvYmktd2FuLmtlbm9iaUBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249Ik9iaS1XYW4gS2Vub2JpIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgIm9yc29uLmtyZW5uaWMiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iT3Jzb24iIC0tc3VybmFtZT0iS3Jlbm5pYyIgLS1pbml0aWFscz0ib2siIC0tbWFpbC1hZGRyZXNzPSJvcnNvbi5rcmVubmljQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iT3Jzb24gS3Jlbm5pYyIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJkYXJ0aC5tYXVsIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkRhcnRoIiAtLXN1cm5hbWU9Ik1hdWwiIC0taW5pdGlhbHM9ImRtIiAtLW1haWwtYWRkcmVzcz0iZGFydGgubWF1bEBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkRhcnRoIE1hdWwiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAibWVsc2hpIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9Ik1lbHNoaSIgLS1pbml0aWFscz0ibSIgLS1tYWlsLWFkZHJlc3M9Im1lbHNoaS5Ac3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJNZWxzaGkiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAibW9uLm1vdGhtYSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJNb24iIC0tc3VybmFtZT0iTW90aG1hIiAtLWluaXRpYWxzPSJtbSIgLS1tYWlsLWFkZHJlc3M9Im1vbi5tb3RobWFAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJNb24gTW90aG1hIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgIm5pZW4ubnVuYiIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJOaWVuIiAtLXN1cm5hbWU9Ik51bmIiIC0taW5pdGlhbHM9Im5uIiAtLW1haWwtYWRkcmVzcz0ibmllbi5udW5iQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iTmllbiBOdW5iIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImxlaWEub3JnYW5hIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkxlaWEiIC0tc3VybmFtZT0iT3JnYW5hIiAtLWluaXRpYWxzPSJsbyIgLS1tYWlsLWFkZHJlc3M9ImxlaWEub3JnYW5hQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iTGVpYSBPcmdhbmEiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiZGFydGguc2lkaW91cyIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJTaGVldiIgLS1zdXJuYW1lPSJQYWxwYXRpbmUiIC0taW5pdGlhbHM9InNwIiAtLW1haWwtYWRkcmVzcz0iZGFydGguc2lkaW91c0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkRhcnRoIFNpZGlvdXMiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAia3lsby5yZW4iIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iQmVuIiAtLXN1cm5hbWU9IlNvbG8iIC0taW5pdGlhbHM9ImJzIiAtLW1haWwtYWRkcmVzcz0ia3lsby5yZW5Ac3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJLeWxvIFJlbiIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJyZXkiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iUmV5IiAtLWluaXRpYWxzPSJyIiAtLW1haWwtYWRkcmVzcz0icmV5LkBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IlJleSIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJib2RoaS5yb29rIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkJvZGhpIiAtLXN1cm5hbWU9IlJvb2siIC0taW5pdGlhbHM9ImJyIiAtLW1haWwtYWRkcmVzcz0iYm9kaGkucm9va0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkJvZGhpIFJvb2siCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiZGFydGgudmFkZXIiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iQW5ha2luIiAtLXN1cm5hbWU9IlNreXdhbGtlciIgLS1pbml0aWFscz0iYXMiIC0tbWFpbC1hZGRyZXNzPSJkYXJ0aC52YWRlckBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkRhcnRoIFZhZGVyIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImx1a2Uuc2t5d2Fsa2VyIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9Ikx1a2UiIC0tc3VybmFtZT0iU2t5d2Fsa2VyIiAtLWluaXRpYWxzPSJscyIgLS1tYWlsLWFkZHJlc3M9Imx1a2Uuc2t5d2Fsa2VyQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iTHVrZSBTa3l3YWxrZXIiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAic3VwcmVtZS1sZWFkZXIuc25va2UiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iU3VwcmVtZS1MZWFkZXIiIC0tc3VybmFtZT0iU25va2UiIC0taW5pdGlhbHM9InNscyIgLS1tYWlsLWFkZHJlc3M9InN1cHJlbWUtbGVhZGVyLnNub2tlQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iU3VwcmVtZSBMZWFkZXIgU25va2UiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiaGFuLnNvbG8iIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iSGFuIiAtLXN1cm5hbWU9IlNvbG8iIC0taW5pdGlhbHM9ImhzIiAtLW1haWwtYWRkcmVzcz0iaGFuLnNvbG9Ac3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJIYW4gU29sbyIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJncmFuZC1tb2ZmLnRhcmtpbiIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJHcmFuZCBNb2ZmIiAtLXN1cm5hbWU9IlRhcmtpbiIgLS1pbml0aWFscz0iZ210IiAtLW1haWwtYWRkcmVzcz0iZ3JhbmQtbW9mZi50YXJraW5Ac3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJHcmFuZCBNb2ZmIFRhcmtpbiIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJyb3NlLnRpY28iIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iUm9zZSIgLS1zdXJuYW1lPSJUaWNvIiAtLWluaXRpYWxzPSJydCIgLS1tYWlsLWFkZHJlc3M9InJvc2UudGljb0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IlJvc2UgVGljbyIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJxdWlubGFuLnZvcyIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJRdWlubGFuIiAtLXN1cm5hbWU9IlZvcyIgLS1pbml0aWFscz0icXYiIC0tbWFpbC1hZGRyZXNzPSJxdWlubGFuLnZvc0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IlF1aW5sYW4gVm9zIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgIndpY2tldC53YXJyaWNrIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IldpY2tldCIgLS1zdXJuYW1lPSJXYXJyaWNrIiAtLWluaXRpYWxzPSJ3d3ciIC0tbWFpbC1hZGRyZXNzPSJ3aWNrZXQud2Fycmlja0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IldpY2tldCBXLiBXYXJyaWNrIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgIndhdHRvIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IldhdHRvIiAtLWluaXRpYWxzPSJ3IiAtLW1haWwtYWRkcmVzcz0id2F0dG8uQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iV2F0dG8iCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAibWFjZS53aW5kdSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJNYWNlIiAtLXN1cm5hbWU9IldpbmR1IiAtLWluaXRpYWxzPSJtdyIgLS1tYWlsLWFkZHJlc3M9Im1hY2Uud2luZHVAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJNYWNlIFdpbmR1IgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgInlvZGEiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iWW9kYSIgLS1pbml0aWFscz0ieSIgLS1tYWlsLWFkZHJlc3M9InlvZGEuQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iWW9kYSIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJjYWQuYmFuZSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJDYWQiIC0tc3VybmFtZT0iQmFuZSIgLS1pbml0aWFscz0iY2IiIC0tbWFpbC1hZGRyZXNzPSJjYWQuYmFuZUBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkNhZCBCYW5lIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImV6cmEuYnJpZGdlciIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJFenJhIiAtLXN1cm5hbWU9IkJyaWRnZXIiIC0taW5pdGlhbHM9ImViIiAtLW1haWwtYWRkcmVzcz0iZXpyYS5icmlkZ2VyQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iRXpyYSBCcmlkZ2VyIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImNhcmEuZHVuZSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJDYXJhIiAtLXN1cm5hbWU9IkR1bmUiIC0taW5pdGlhbHM9ImNkIiAtLW1haWwtYWRkcmVzcz0iY2FyYS5kdW5lQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQ2FyYSBEdW5lIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImdhcnNhLmZ3aXAiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iR2Fyc2EiIC0tc3VybmFtZT0iRndpcCIgLS1pbml0aWFscz0iZ2YiIC0tbWFpbC1hZGRyZXNzPSJnYXJzYS5md2lwQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iR2Fyc2EgRndpcCIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJtb2ZmLmdpZGVvbiIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJNb2ZmIiAtLXN1cm5hbWU9IkdpZGVvbiIgLS1pbml0aWFscz0ibWciIC0tbWFpbC1hZGRyZXNzPSJtb2ZmLmdpZGVvbkBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249Ik1vZmYgR2lkZW9uIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImJhYnkueW9kYSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJEaW4iIC0tc3VybmFtZT0iR3JvZ3UiIC0taW5pdGlhbHM9ImRnIiAtLW1haWwtYWRkcmVzcz0iYmFieS55b2RhQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQmFieSBZb2RhIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImthbmFuLmphcnJ1cyIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJDYWxlYiIgLS1zdXJuYW1lPSJEdW1lIiAtLWluaXRpYWxzPSJjZCIgLS1tYWlsLWFkZHJlc3M9ImthbmFuLmphcnJ1c0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkthbmFuIEphcnJ1cyIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJncmVlZi5rYXJnYSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJHcmVlZiIgLS1zdXJuYW1lPSJLYXJnYSIgLS1pbml0aWFscz0iZ2siIC0tbWFpbC1hZGRyZXNzPSJncmVlZi5rYXJnYUBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkdyZWVmIEthcmdhIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImJvLWthdGFuLmtyeXplIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkJvLUthdGFuIiAtLXN1cm5hbWU9IktyeXplIiAtLWluaXRpYWxzPSJiayIgLS1tYWlsLWFkZHJlc3M9ImJvLWthdGFuLmtyeXplQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQm8tS2F0YW4gS3J5emUiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAia3VpaWwiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iS3VpaWwiIC0taW5pdGlhbHM9ImsiIC0tbWFpbC1hZGRyZXNzPSJrdWlpbC5Ac3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJLdWlpbCIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJkaW4uZGphcmluIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkRpbiIgLS1zdXJuYW1lPSJEamFyaW4iIC0taW5pdGlhbHM9ImRkIiAtLW1haWwtYWRkcmVzcz0iZGluLmRqYXJpbkBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IlRoZSBNYW5kYWxvcmlhbiIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJtaWdzLm1heWZlbGQiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iTWlncyIgLS1zdXJuYW1lPSJNYXlmZWxkIiAtLWluaXRpYWxzPSJtbSIgLS1tYWlsLWFkZHJlc3M9Im1pZ3MubWF5ZmVsZEBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249Ik1pZ3MgTWF5ZmVsZCIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJjdC03NTY3IiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkNhcHRhaW4iIC0tc3VybmFtZT0iUmV4IiAtLWluaXRpYWxzPSJjciIgLS1tYWlsLWFkZHJlc3M9ImN0LTc1NjdAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJDVC03NTY3IgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImZlbm5lYy5zaGFuZCIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJGZW5uZWMiIC0tc3VybmFtZT0iU2hhbmQiIC0taW5pdGlhbHM9ImZzIiAtLW1haWwtYWRkcmVzcz0iZmVubmVjLnNoYW5kQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iRmVubmVjIFNoYW5kIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImhlcmEuc3luZHVsbGEiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iSGVyYSIgLS1zdXJuYW1lPSJTeW5kdWxsYSIgLS1pbml0aWFscz0iaHMiIC0tbWFpbC1hZGRyZXNzPSJoZXJhLnN5bmR1bGxhQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iSGVyYSBTeW5kdWxsYSIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJhaHNva2EudGFubyIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJBaHNva2EiIC0tc3VybmFtZT0iVGFubyIgLS1pbml0aWFscz0iYXQiIC0tbWFpbC1hZGRyZXNzPSJhaHNva2EudGFub0BzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkFoc29rYSBUYW5vIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImFzYWpqLnZlbnRyZXNzIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkFzYWpqIiAtLXN1cm5hbWU9IlZlbnRyZXNzIiAtLWluaXRpYWxzPSJhdiIgLS1tYWlsLWFkZHJlc3M9ImFzYWpqLnZlbnRyZXNzQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQXNhamogVmVudHJlc3MiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAicGF6LnZpenNsYSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJQYXoiIC0tc3VybmFtZT0iVml6c2xhIiAtLWluaXRpYWxzPSJwdiIgLS1tYWlsLWFkZHJlc3M9InBhei52aXpzbGFAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJQYXogVml6c2xhIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImNhbC5rZXN0aXMiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iQ2FsIiAtLXN1cm5hbWU9Iktlc3RpcyIgLS1pbml0aWFscz0iY2siIC0tbWFpbC1hZGRyZXNzPSJjYWwua2VzdGlzQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQ2FsIEtlc3RpcyIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJyZXZhbiIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJSZXZhbiIgLS1pbml0aWFscz0iciIgLS1tYWlsLWFkZHJlc3M9InJldmFuLkBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IlJldmFuIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImlkZW4udmVyc2lvIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IklkZW4iIC0tc3VybmFtZT0iVmVyc2lvIiAtLWluaXRpYWxzPSJpdiIgLS1tYWlsLWFkZHJlc3M9ImlkZW4udmVyc2lvQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iSWRlbiBWZXJzaW8iCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiZGFydGguYmFuZSIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJEYXJ0aCIgLS1zdXJuYW1lPSJCYW5lIiAtLWluaXRpYWxzPSJkYiIgLS1tYWlsLWFkZHJlc3M9ImRhcnRoLmJhbmVAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJEYXJ0aCBCYW5lIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImJsYWNrLmtycnNhbnRhbiIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJCbGFjayIgLS1zdXJuYW1lPSJLcnJzYW50YW4iIC0taW5pdGlhbHM9ImJrIiAtLW1haWwtYWRkcmVzcz0iYmxhY2sua3Jyc2FudGFuQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQmxhY2sgS3Jyc2FudGFuIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImRhcnRoLnBsYWd1ZWlzIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkRhcnRoIiAtLXN1cm5hbWU9IlBsYWd1ZWlzIiAtLWluaXRpYWxzPSJkcCIgLS1tYWlsLWFkZHJlc3M9ImRhcnRoLnBsYWd1ZWlzQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iRGFydGggUGxhZ3VlaXMiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAicmVuIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IlJlbiIgLS1pbml0aWFscz0iciIgLS1tYWlsLWFkZHJlc3M9InJlbi5Ac3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJSZW4iCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAibWl0dGhyYXdudXJ1b2RvIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkdyYW5kLUFkbWlyYWwiIC0tc3VybmFtZT0iVGhyYXduIiAtLWluaXRpYWxzPSJnYXQiIC0tbWFpbC1hZGRyZXNzPSJtaXR0aHJhd251cnVvZG9Ac3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJHcmFuZCBBZG1pcmFsIFRocmF3biIKc3VkbyBzYW1iYS10b29sIHVzZXIgY3JlYXRlICJjb2JiLnZhbnRoIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkNvYmIiIC0tc3VybmFtZT0iVmFudGgiIC0taW5pdGlhbHM9ImN2IiAtLW1haWwtYWRkcmVzcz0iY29iYi52YW50aEBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkNvYmIgVmFudGgiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiZGFydGguY2FlZHVzIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkphY2VuIiAtLXN1cm5hbWU9IlNvbG8iIC0taW5pdGlhbHM9ImpzIiAtLW1haWwtYWRkcmVzcz0iZGFydGguY2FlZHVzQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iRGFydGggQ2FlZHVzIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImt5bGUua2F0YXJuIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9Ikt5bGUiIC0tc3VybmFtZT0iS2F0YXJuIiAtLWluaXRpYWxzPSJrayIgLS1tYWlsLWFkZHJlc3M9Imt5bGUua2F0YXJuQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iS3lsZSBLYXRhcm4iCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiZGFydGgudHJheWEiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iS3JlaWEiIC0taW5pdGlhbHM9ImsiIC0tbWFpbC1hZGRyZXNzPSJkYXJ0aC50cmF5YUBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IkRhcnRoIFRyYXlhIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImNhcnRoLm9uYXNpIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9IkNhcnRoIiAtLXN1cm5hbWU9Ik9uYXNpIiAtLWluaXRpYWxzPSJjbyIgLS1tYWlsLWFkZHJlc3M9ImNhcnRoLm9uYXNpQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQ2FydGggT25hc2kiCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAiYXR0b24ucmFuZCIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJBdHRvbiIgLS1zdXJuYW1lPSJSYW5kIiAtLWluaXRpYWxzPSJhciIgLS1tYWlsLWFkZHJlc3M9ImF0dG9uLnJhbmRAc3RhcndhcnMuY29tIiAtLWRlc2NyaXB0aW9uPSJBdHRvbiBSYW5kIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgImJhc3RpbGEuc2hhbiIgVWJ1bnR1MSsgLS1naXZlbi1uYW1lPSJCYXN0aWxhIiAtLXN1cm5hbWU9IlNoYW4iIC0taW5pdGlhbHM9ImJzIiAtLW1haWwtYWRkcmVzcz0iYmFzdGlsYS5zaGFuQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iQmFzdGlsYSBTaGFuIgpzdWRvIHNhbWJhLXRvb2wgdXNlciBjcmVhdGUgIm1pc3Npb24udmFvIiBVYnVudHUxKyAtLWdpdmVuLW5hbWU9Ik1pc3Npb24iIC0tc3VybmFtZT0iVmFvIiAtLWluaXRpYWxzPSJtdiIgLS1tYWlsLWFkZHJlc3M9Im1pc3Npb24udmFvQHN0YXJ3YXJzLmNvbSIgLS1kZXNjcmlwdGlvbj0iTWlzc2lvbiBWYW8iCnN1ZG8gc2FtYmEtdG9vbCB1c2VyIGNyZWF0ZSAidmV0dGUiIFVidW50dTErIC0tZ2l2ZW4tbmFtZT0iQ8OpbmEiIC0taW5pdGlhbHM9ImMiIC0tbWFpbC1hZGRyZXNzPSJ2ZXR0ZUBzdGFyd2Fycy5jb20iIC0tZGVzY3JpcHRpb249IlZldHRlIgoK
        owner: 'root:root'
        path: /usr/local/bin/add-samba-users.sh
        permissions: '0750'
      - encoding: b64
        content: IyEvYmluL2Jhc2gKc3VkbyBzYW1iYS10b29sIHVzZXIgbGlzdHxhd2sgJyEva3JidGd0fEd1ZXN0fEFkbWluaXN0cmF0b3Ive3ByaW50ICJceDIyIiQxIlx4MjIifSd8eGFyZ3MgLXJuMSAtUDAgYmFzaCAtYyAnc3VkbyBzYW1iYS10b29sIHVzZXIgZGVsZXRlICQwJwo=
        owner: 'root:root'
        path: /usr/local/bin/del-samba-users.sh
        permissions: '0750'
    resolv_conf:
      nameservers: ['${DNS1}', ${DNS2}', '${DNS_FWDER}']
      searchdomains: [${DNS_SEARCH}]
      domain: ${DNS_DOMAIN}
      options:
        rotate: true
        timeout: 1
    timezone: ${TZ}
    locale: ${LOCALE}
    groups:
      - ubuntu
      - demo
    users:
      - name: ubuntu
        homedir: /home/ubuntu
        gecos: Default User
        groups: [ubuntu, adm, audio, cdrom, dialout, dip, floppy, lxd, netdev, plugdev, sudo, video]
        primary_group: ubuntu
        lock_passwd: false
        # Salted PW = `echo -n ubuntu |mkpasswd --method=SHA-512 --rounds=4096 -s`
        passwd: \$6$rounds=4096$ox6T7Xv0j9sYJhd7$VIw3A8RVAHAP/vfZFJFNOupES3IqL4M64TjHTKYNmCAiNzZN0I3hdLGYGj7ppFYU0Nzc6Wn7EgkyKzK.afkBB0
        sudo: ALL=(ALL) NOPASSWD:ALL
        shell: /bin/bash
        ssh_import_id: [${SSH_IMPORT_ID}]
      - name: demo
        homedir: /home/demo
        gecos: Demo User
        primary_group: demo
        groups: [demo, adm, audio, cdrom, dialout, dip, floppy, lxd, netdev, plugdev, sudo, video]
        lock_passwd: false
        # Salted PW = `echo -n demo |mkpasswd --method=SHA-512 --rounds=4096 -s`
        passwd: \$6$rounds=4096$e2ZBaCaw7Uxc5DW0$3nvsEyNMTAoRp6PhLRbm0BoGfgUVXEjgQqqXcZxIhz2EsTNsEeERdIzC1wbXGFvt2LN3gc0C7KarDD9KWiuxK1
        sudo: ALL=(ALL) NOPASSWD:ALL
        shell: /bin/bash
        ssh_import_id: [${SSH_IMPORT_ID}]
    package_update: yes
    package_upgrade: yes
    packages: [acl, apt-transport-https, apt-utils, attr, build-essential, ca-certificates, chrony, curl, debconf-utils, dnsutils, git, gnupg, jq, krb5-config, krb5-user, libnss-winbind, libpam-krb5, libpam-winbind, make, net-tools, openssl, python3-apt, python3-pip, python3-setproctitle, samba, samba-dsdb-modules, samba-vfs-modules, smbclient, software-properties-common, unzip, vim, wget, whois, winbind]
    apt:
      conf: |
        APT {
          Get {
            Assume-Yes True;
            Fix-Broken True;
          };
          Acquire {
            ForceIPv4 True;
          };
        };
      primary:
       - arches: [amd64]
         uri: 'http://${UBUNTU_REPO}/ubuntu/'
      security:
       - arches: [amd64]
         uri: 'http://${UBUNTU_REPO}/ubuntu/'
      sources_list: |
        deb [arch=amd64] \$PRIMARY \$RELEASE main universe restricted multiverse
        deb [arch=amd64] \$PRIMARY \$RELEASE-updates main universe restricted multiverse
        deb [arch=amd64] \$SECURITY \$RELEASE-security main universe restricted multiverse
        deb [arch=amd64] \$PRIMARY \$RELEASE-backports main universe restricted multiverse
    ssh_authorized_keys:
      - ecdsa-sha2-nistp521 AAAAE2VjZHNhLXNoYTItbmlzdHA1MjEAAAAIbmlzdHA1MjEAAACFBADviK4QkET0s1TxcPH0ezmdLcAtlyvsM1kN5mYkupzoHuscB5cw6rU6MoHVylwzj41/U2zJYFGoWLOCALEahyg/dfpNQBqep0OdxcDm3aBnswD+Vac49zmOo56cNOJeluPIiHyIF3ys6k3NEGW9sBdNFMVFs4RX8SurFvPTqMSoQoSJ4PQ8Q== craigbender@canonical.com
      - ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCcHM9zZP5Ca00FKtNDLk+PXXKeHJjgzGhNoMxKGWUDxq46ei5J0Bwz5G0zya+H1KbDNowBO4Az0cXWV3Zyq+m3KRamQdGH6rmEH9M7v8+OMdD9biJhWVhEOXfB0tSyxTjoipRTkyLdGZRdZ+o0Af7OxNx21Eo84QDR2H+4cBLwFA8l7yFJrY8aR0dPsWMcMBEdTydH13LvMV/dII1J6Fppfi+eDOoy8HpnlAs3411QNgR1IQow3vqpynnkaH68oRi2Db0bOQC6EUe2mCRVqI5Ro4OgtS9JlZJZ8BxikkyxujapH9K3xZYl6HG4lq7WWYIme4uMM2xo8rLMwfWytyjNfWJfRmNsxtUGywBQdIipe2FE7F05nPClmb4U2B5rAJiNjTJNCnhiZMaaF1C8kVExf4ldarZMTBHfQAoDizHrn6m4VPpVKCMM4zuc177QxPPtHSDMpgt2KXegJfXaU3UW4xc0aH8yrCX+4QPe9yQQ464edGf5iLwonheUVXxf58v3yVCDS3b7CBKpgU0xOIcsx8IPYkWfHKlBwtpZR1JVV0LiW9ivXyJJgQLOUGVQ70FeVx+uLT+HuWLc4rVLmzHMBJhpS+cEMGBnOSu5IXYfK2n4v1MrQBMS13SA6NwxZ15mf5FKs0oxFFk3qERTQl9+FhzGjwHq9vojX0vXXyML8w== craigbender@canonical.com
    bootcmd:
      - ['cloud-init-per', 'once', 'msg0', 'sh', '-c', 'echo "\e[1;48;2;0;255;0m\e[1;38;2;0;0;0m=========Starting Cloud-Init BOOTCMDs=========\e[0m"']
      - ['cloud-init-per', 'once', 'env0', 'set', '-x']
      - ['cloud-init-per', 'once', 'env1', 'export', 'DEBIAN_FRONTEND=noninteractive']
      - ['cloud-init-per', 'once', 'env2', 'cloud-init', 'schema', '--system']
      - ['cloud-init-per', 'once', 'apt0', '/usr/bin/apt-get', '--option=Acquire::ForceIPv4=true', 'update']
      - ['cloud-init-per', 'once', 'apt1', '/usr/bin/apt-get', '--option=Acquire::ForceIPv4=true', 'install', '--auto-remove', '--purge', '-fy']
      - ['cloud-init-per', 'once', 'msg1', 'sh', '-c', 'echo "\e[1;48;2;0;255;0m\e[1;38;2;0;0;0m=========Finished Cloud-Init BOOTCMDs=========\e[0m"']
    runcmd:
      - sh -c 'echo "\e[1;48;2;0;255;0m\e[1;38;2;0;0;0m=========Starting Cloud-Init RUNCMDs=========\e[0m"'
      - set -x
      - export DEBIAN_FRONTEND=noninteractive
      # General Setup
      - update-alternatives --set editor /usr/bin/vim.basic
      - export DEFAULT_CIDR=\$(ip -o -4 a show dev \$(ip -o route show default|grep -m1 -oP "(?<=dev )[^ ]+")|grep -m1 -oP "(?<=inet )[^ ]+")
      - export DEFAULT_IP=\$(ip -o -4 a show dev \$(ip -o route show default|grep -m1 -oP "(?<=dev )[^ ]+")|grep -m1 -oP "(?<=inet )[^/]+")
      - if \$(test -f /etc/fuse.conf);then sed -i 's/^#user_allow/user_allow/g' /etc/fuse.conf;fi;
      - if ! \$(grep -qE '^user_allow_other' /etc/fuse.conf);then sed -i '\$auser_allow_other' /etc/fuse.conf;fi;
      - systemctl restart systemd-networkd systemd-resolved
      - su - \$(id -un 1000) -c 'printf "y\n"|ssh-keygen -t rsa -b 4096 -f /home/\$(id -un 1000)/.ssh/id_rsa -P ""'
      - su - \$(id -un 1000) -c 'printf "y\n"|ssh-keygen -t ecdsa -b 521 -f /home/\$(id -un 1000)/.ssh/id_ecdsa -P ""'
      - su - \$(id -un 1000) -c 'printf "y\n"|ssh-keygen -t dsa -b 1024 -f /home/\$(id -un 1000)/.ssh/id_dsa -P ""'
      - su - \$(id -un 1000) -c 'printf "y\n"|ssh-keygen -t ed25519 -f /home/\$(id -un 1000)/.ssh/id_ed25519 -P ""'
      - su - \$(id -un 1000) -c 'cat ~/.ssh/*.pub|tee 1>/dev/null -a ~/.ssh/authorized_keys'
      # sudoers prefs for which env variables to retain when sudoing
      - |-
        cat <<CISUDOERS |sed -r 's/[ \t]+$//g'|tee -a /etc/sudoers.d/99-defaults
        Defaults\$(printf "\t")env_keep+="DEFAULT_* PG* MAAS* RBAC* CANDID* LDS* SSP* DISPLAY EDITOR HOME LANG* LC* PS* *_IP *_PROXY *_proxy"
        Defaults\$(printf "\t")secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:\\\\$HOME/.local/bin"
        CISUDOERS
      - set +x
      - sh -c 'echo "\e[1;48;2;233;84;20m\e[1;38;2;255;255;255m=========SAMBA Setup - Starting=========\e[0m"'
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - krb5 debconf=========\e[0m"'
      - set -x
      # Configure Kerberos for our Domain
      - set -x
      - printf "krb5-config\tkrb5-config/add_servers\tboolean\ttrue\nkrb5-config\tkrb5-config/add_servers_realm\tstring\t\$(hostname -d|awk '{print toupper(\$0)}')\nkrb5-config\tkrb5-config/admin_server\tstring\t\$(hostname -f)\nkrb5-config\tkrb5-config/default_realm\tstring\t\$(hostname -d|awk '{print toupper(\$0)}')\nkrb5-config\tkrb5-config/kerberos_servers\tstring\t\$(hostname -f)\nkrb5-config\tkrb5-config/read_conf\tboolean\ttrue\n"|debconf-set-selections
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - /etc/hosts =========\e[0m"'
      - set -x
      # DNS Setup for SAMBA AD
      - sed -r -i "/0.1.1/d;/^127.*localhost/a \$DEFAULT_IP\t\$(hostname -f) \$(hostname -s)\n\$DEFAULT_IP\t\$(hostname -d)\n" /etc/hosts
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - DNS (Static Resolver)=========\e[0m"'
      - set -x
      - systemctl disable --now systemd-resolved
      - unlink /etc/resolv.conf
      - rm -rf /etc/resolv.conf
      - install -o\$(id -un 0) -g\$(id -gn 0) -m0644 /dev/null /etc/resolv.conf
      - |-
        cat <<RESOLV |sed -r 's/[ \t]+$//g'|tee /etc/resolv.conf
        # Samba server IP address
        nameserver \$DEFAULT_IP
        # fallback resolver
        nameserver ${DNS_FWDER}
        # main domain for Samba
        search \$(hostname -f)
        RESOLV
      # Disable SAMBA Services that conflict with SAMBA AD Role of "dc"
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Disable services smbd, nmbd, winbind=========\e[0m"'
      - set -x
      - sudo systemctl disable --now smbd nmbd winbind
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Enable service samba-ad-dc=========\e[0m"'
      - set -x
      # Unmask and Enable SAMBA AD Domain Controller Service
      - systemctl unmask samba-ad-dc
      - systemctl enable samba-ad-dc
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Purge existing smb.conf=========\e[0m"'
      - set -x
      # Remove stock smb.conf, otherwise domain provisioning will fail
      - rm -rf /etc/samba/smb.conf
      # Max UID in unprivileged container is  65534.  Samba attempts to use 3000000+, fix that here.
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Change UID range to < 65534=========\e[0m"'
      - set -x
      - |-
        cat <<UIDEOF |sed -r 's/[ \t]+$//g'|tee /usr/share/samba/setup/idmap_init.ldif
        dn: CN=CONFIG
        cn: CONFIG
        lowerBound: 655
        upperBound: 65533

        dn: @INDEXLIST
        @IDXATTR: xidNumber
        IDXATTR: objectSid

        dn: CN=S-1-5-32-544
        cn: S-1-5-32-544
        objectClass: sidMap
        objectSid: S-1-5-32-544
        type: ID_TYPE_BOTH
        xidNumber: 655
        distinguishedName: CN=S-1-5-32-544
        UIDEOF
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Provision AD Domain \$(hostname -d)=========\e[0m"'
      - set -x
      - "samba-tool domain provision --server-role=dc --use-rfc2307 --dns-backend=SAMBA_INTERNAL --realm=${DNS_DOMAIN^^} --domain=${MSAD_DOMAIN^^} --adminpass='${MSADMIN_PW}' --krbtgtpass='${KRBTGT_PW}' --option='vfs objects = acl_xattr xattr_tdb' --option='idmap config * : range = 655-65533' --option='dns forwarder = ${DNS_FWDER}'"
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Move in SAMBA generated krb5.conf=========\e[0m"'
      - set -x
      - cp /var/lib/samba/private/krb5.conf /etc/krb5.conf
      - systemctl start samba-ad-dc
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Configure Chrony (NTP)=========\e[0m"'
      - set -x
      - chown root:_chrony /var/lib/samba/ntp_signd
      - chmod 750 /var/lib/samba/ntp_signd
      - |-
        cat <<NTPEOF |sed -r 's/[ \t]+$//g'|tee -a /etc/chrony/chrony.conf
        # bind the chrony service to IP address of the Samba AD
        bindcmdaddress \$DEFAULT_IP
        # allow clients on the network to connect to the Chrony NTP server
        allow 0.0.0.0/0
        # specify the ntpsigndsocket directory for the Samba AD
        ntpsigndsocket /var/lib/samba/ntp_signd
        NTPEOF
      - systemctl restart chronyd
      - systemctl status chronyd -l --no-pager
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Name resolution Testing=========\e[0m"'
      - set -x
      - host -t A \$(hostname -d)
      - host -t A \$(hostname -f)
      - host -t SRV _kerberos._udp.\$(hostname -d)
      - host -t SRV _kerberos._tcp.\$(hostname -d)
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - krb5 init=========\e[0m"'
      - set -x
      - echo '${MSADMIN_PW}'|kinit administrator
      - klist
      - set +x
      - sh -c 'echo "\e[1;38;2;233;84;20m\e[1;48;2;255;255;255m=========SAMBA Setup - Adding (19)77 users to Active Directory=========\e[0m"'
      - set -x
      - if \$(test -f /usr/local/bin/add-samba-users.sh);then /usr/local/bin/add-samba-users.sh;fi
      - set +x
      - sh -c 'echo "\e[1;48;2;233;84;20m\e[1;38;2;255;255;255m=========SAMBA Setup - Finished=========\e[0m"'
      - sh -c 'echo "\e[1;48;2;0;255;0m\e[1;38;2;0;0;0m=========Finished Cloud-Init RUNCMDs=========\e[0m"'
description: MSAD Controller for Canonical Candid/RBAC Demos
devices:
  ${LXD_CHILD_NIC}:
    name: ${LXD_CHILD_NIC}
    nictype: bridged
    parent: ${LXD_PARENT_NIC}
    type: nic
  root:
    path: /
    pool: ${LXD_STORAGE_POOL}
    size: 10GiB
    type: disk
name: ${LXD_PROFILE_NAME}
used_by: []
LXDPROF

lxc 2>/dev/null delete ${DC_HOSTNAME} -f;
lxc launch ubuntu-daily:${UBUNTU_RELEASE} ${DC_HOSTNAME} -p ${LXD_PROFILE_NAME} --console;
