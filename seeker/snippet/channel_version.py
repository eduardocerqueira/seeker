#date: 2022-04-26T17:15:51Z
#url: https://api.github.com/gists/0a0e67b4518618c2b1471c43e86e9270
#owner: https://api.github.com/users/fushall

from bs4 import BeautifulSoup as bs
import urllib3
import json
import threading
import time

def get_html(url):
    http = urllib3.PoolManager()
    response = http.request('GET', url, timeout=4.0)
    content = response.data
    soup = bs(content, 'html.parser')
    return soup


def get_data(url):
    http = urllib3.PoolManager()
    response = http.request('GET', url, timeout=4.0)
    return response.data


def get_json(url):
    return json.loads(get_data(url).decode('utf-8'))


# def get_protobuf2json(url):
#     return get_data(url)

def get_yyb():
    # print('yyb', time.time())
    yyb = get_html('https://a.app.qq.com/o/simple.jsp?pkgname=com.banma')
    version = yyb.find_all(class_='pp-comp-extra-p')[1].string.replace('\n', '').replace('\r', '')
    print('应用宝   '+version)

def get_mi():
    # print('mi', time.time())
    mi = get_json('https://m.app.mi.com/detailapi/730365')
    print('小  米   版本：'+mi['appMap']['vname'])  

def get_hw():
    # print('hw', time.time())
    hw = get_json('https://web-drcn.hispace.dbankcloud.cn/uowap/index?method=internal.getTabDetail&uri=app|C100509837&appid=C100509837')
    print('华  为   版本：'+hw['layoutData'][10]['dataList'][0]['versionName'])

def get_vivo():
    # print('vivo', time.time())
    vivo = get_html('http://info.appstore.vivo.com.cn/detail/2395080')
    version = vivo.select('li[class="ly-com.github.mybatis.fl"]')[0].string
    print(' VIVO    '+version)

# oppo使用protobuf难以解析
# feed.ParseFromString(urllib3.PoolManager().request('GET', 'https://api-cn.store.heytapmobi.com/detail/v4/ext-infos?install=0&appid=3719450&size=10&start=0', timeout=4.0).read())
# oppo = get_protobuf2json('https://api-cn.store.heytapmobi.com/detail/v4/ext-infos?install=0&appid=3719450&size=10&start=0')
# print(feed)
# print('OPPO   版本：'+hw['layoutData'][10]['dataList'][0]['versionName'])
start_time = time.time()
list=[threading.Thread(target=get_yyb),threading.Thread(target=get_mi),threading.Thread(target=get_hw),threading.Thread(target=get_vivo)]
for t in list:
    # t.setDaemon(True)
    t.start()
for t in list:
    t.join()
# get_mi()
# get_yyb()
# get_hw()
# get_vivo()
print('主线程结束了！' , threading.current_thread().name)
print('一共用时：', time.time()-start_time)