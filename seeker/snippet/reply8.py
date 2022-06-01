#date: 2022-06-01T17:02:26Z
#url: https://api.github.com/gists/7d2c77d19af56f92540545056285d082
#owner: https://api.github.com/users/Zhenger233

import requests
import hashlib
import random
import time
import json
import sys

def get_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def getapphash():
    s = f'{time.time()}'[:5] + 'appbyme_key'
    apphash = get_md5(s)
    ans = apphash[8:16]
    return ans

def getInfo(key):
    info = json.load(open('info.json', 'r', encoding='utf-8'))
    return info[key]

token = getInfo('token')
secret = getInfo('secret')
urlBase = 'http://bbs.uestc.edu.cn/mobcent/app/web/index.php'
tid = 1940803
contentfile = '新手导航.txt'

paramst = {
    'r': 'forum/postlist',
    'topicId': tid,
    'pageSize': 1,
    'page': 1,
    'order': 1,
    'apphash': getapphash(),
    'accessToken': token,
    'accessSecret': secret
}

paramsl = {
    'r': 'user/login',
    'type': 'login',
    'username': getInfo('username'),
    'password': getInfo('password'),
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
    'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
}

def getParamsReply(tid, content):
    replycontent = [{'type': 0,'infor': content}]
    replyjson = {'body': {'json':{'tid': tid,'content': json.dumps(replycontent)}}}
    return {
        'r': 'forum/topicadmin',
        'act': 'reply',
        'json': json.dumps(replyjson),
        'apphash': getapphash(),
        'accessToken': token,
        'accessSecret': secret
    }

def numo(i, filename = 'number.txt'):
    with open(filename, 'w', encoding = 'utf-8') as f:
        f.write(str(i))

def numi(filename = 'number.txt'):
    with open(filename, 'r', encoding = 'utf-8') as f:
        i = f.read()
    return int(i)

def getFile(filename = '版规.txt', i = 0):
    with open(filename, 'r', encoding = 'utf-8') as f:
        content = f.read().split('\n')
        l = len(content)
    return content[i % l]

def reply8():
    num = numi()
    for i in range(num, num + 8):
        if i >= 124:break
        print(i)
        txt = getFile(filename = contentfile, i = i)
        print(txt)
        if len(txt) < 6:
            txt = txt + '\1\1\1\1\1\1'
        r = requests.post(urlBase, params = getParamsReply(tid,txt), headers = headers)
        if r.status_code == 200:
            numo(i + 1)
            print(r.json())
            time.sleep(4 + random.randint(1, 4))
        else:
            print(r.status_code)
            print(r.text)
            break

def check():
    r = requests.post(urlBase, params = paramst, headers = headers)
    print(r.json()['list'][0]['reply_name'])

def genAccess():
    r = requests.post(urlBase, params = paramsl, headers = headers)
    print(r.json())   
    token = r.json()['token']
    secret = r.json()['secret']
    i = json.load(open('info.json', 'r', encoding = 'utf-8'))
    i['token'] = token
    i['secret'] = secret
    json.dump(i, open('info.json', 'w', encoding = 'utf-8'))
    print(i)

def test():
    print('welcome to fuck hepan!')
    # print(getapphash())
    # reply8()
    # check()
    # genAccess()
    # print(getFile('版规.txt',132))

if __name__ == '__main__':
    test()
    if 'access' in sys.argv:
        genAccess()
    if 'reply' in sys.argv:
        reply8()
    if 'check' in sys.argv:
        check()
