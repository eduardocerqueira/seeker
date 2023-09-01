#date: 2023-09-01T16:52:25Z
#url: https://api.github.com/gists/c8fc857cdd323268cc8ee1042977d92e
#owner: https://api.github.com/users/blossomsg

import shotgun_api3


# create a token and passphrase it is necessary for this process
# https://help.autodesk.com/view/SGSUB/ENU/?guid=SG_Migration_mi_migration_account_mi_end_user_account_html
# https: "**********"
# https://developer.shotgridsoftware.com/python-api/reference.html#shotgun_api3.shotgun.Shotgun.find
SERVER_PATH = 'https://<company>.shotgunstudio.com' #shotgun company link
LOGIN = 'bghuntla'
PASSWORD = "**********"

sg = "**********"=LOGIN, password=PASSWORD)
if __name__ == '__main__':
    filters = [['sg_status_list', 'is', 'act']]
    fields = ['id']
    users = sg.find('HumanUser', filters, fields)

    print('Total active Users:{}'.format(len(users)))tus_list', 'is', 'act']]
    fields = ['id']
    users = sg.find('HumanUser', filters, fields)

    print('Total active Users:{}'.format(len(users)))