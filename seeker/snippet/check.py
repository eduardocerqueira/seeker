#date: 2022-06-30T16:59:05Z
#url: https://api.github.com/gists/23dac9126310d5904e3360137c8fd9e6
#owner: https://api.github.com/users/hclivess

target_list = [{"ip": 1, "ad": 0}, {"ip": 2, "ad": 0}]
s2 = {"ip": 2}
s3 = {"ip": 3}

if s2["ip"] not in [x["ip"] for x in target_list]:
    print(f"{s2['ip']} is new")

if s3["ip"] not in [x["ip"] for x in target_list]:
    print(f"{s2['ip']} is new")