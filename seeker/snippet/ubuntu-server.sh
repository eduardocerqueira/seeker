#date: 2022-03-07T17:11:39Z
#url: https://api.github.com/gists/1dd9cdbda9768511f3871c45b1feb348
#owner: https://api.github.com/users/MikhailKalikin

# /etc/exports: the access control list for filesystems which may be exported
#		to NFS clients.  See exports(5).
#
# Example for NFSv2 and NFSv3:
# /srv/homes       hostname1(rw,sync,no_subtree_check) hostname2(ro,sync,no_subtree_check)
#
# Example for NFSv4:
# /srv/nfs4        gss/krb5i(rw,sync,fsid=0,crossmnt,no_subtree_check)
# /srv/nfs4/homes  gss/krb5i(rw,sync,no_subtree_check)
#

/home/joaomlneto   192.168.25.0/24(rw,sync,no_subtree_check,insecure,no_root_squash,all_squash,anonuid=1000)