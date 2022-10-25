#date: 2022-10-25T17:34:08Z
#url: https://api.github.com/gists/3e2e65bf6395759498dd4485e2a55f62
#owner: https://api.github.com/users/waitusz

# !/usr/bin/env python3
#  -*- coding: utf-8 -*-

import os
import sys
import collections

# 文件内容如下（获得）
# [2018春节爆字统计]用户(123456)获得福字(0)(1)

# 文件内容如下（消耗）
# [2018春节会员兑换统计]用户(888888)进行会员兑换(0), 目前等级(2), 到期时间(1519997460), 兑换前等级(1), 到期时间(1519997456)

file_get = "./TheSpringFestival_get"
file_consum = "./TheSpringFestival_xiao"

# 定义空字典存储数据
dict_get = {}


# 解析每一行数据(获得)
def get_data(tmp_line):
    splist_result = tmp_line.split('(')
    usernum = splist_result[1].split(')')[0]
    key_nums = splist_result[2].split(')')[0]
    key_count = splist_result[3].split(')')[0]
    return usernum, key_nums


# 解析每一行数据(消耗)
def get_data2(tmp_line):
    splist_result = tmp_line.split('(')
    usernum = splist_result[1].split(')')[0]
    return usernum


# 分析判断并保存数据
def func_ana_get():
    print('Enter func!')
    fp_get = open(file_get, "a+")
    get_lins = fp_get.readlines()
    for tmp_line in get_lins:
        # print tmp_line
        user, key = get_data(tmp_line)
        if dict_get.has_key(user):
            if dict_get[user].has_key(key):
                dict_get[user][key] = dict_get[user][key] + 1
            else:
                dict_get[user][key] = 1
                pass
        else:
            tmp_dict = {}
            tmp_dict[key] = 1
            dict_get[user] = tmp_dict
            fp_get.close()


# 消耗
def func_ana_xiao():
    print("Enter func!")
    fp_get = open(file_consum, "a+")
    get_lins = fp_get.readlines()
    for tmp_line in get_lins:
        # print tmp_line.strip()
        user = get_data2(tmp_line)
        if dict_get.has_key(user):
            for keys in dict_get[user]:
                dict_get[user][keys] = dict_get[user][keys] - 1
        else:
            pass
    fp_get.close()


# 统计字典中各值的和
def get_sum(user_data):
    sum = 0
    for keys in user_data:
        sum = sum + user_data[keys]
    return sum


# 统计排序
def func_ana_tonji():
    sort_dict = {}
    for keys in dict_get:
        sum = get_sum(dict_get[keys])
        sort_dict[keys] = sum

        aa = sorted(sort_dict.items(), key=lambda item: item[1], reverse=True)
        print(aa)
        nn = 0
    for keys in aa:
        print(keys[0], keys[1])
        if nn > 10:
            break
        nn = nn + 1


func_ana_get()
func_ana_xiao()
func_ana_tonji()

#  -----------------------------------
# !/usr/bin/env python
# --coding:utf-8--
# 会员鲜花库存统计
import MySQLdb
import os, sys, re, string
import time, getopt

optmap = {
    'dbuser': 'haoren',
    'dbpass': 'ddddd',
    'dbhost': '172.17.1.14',
    'dbhost_gm': '172.17.1.13',
    'dbport': 3306,
    'dbname': 'PIWMDB',
    'dbname_gm': 'TGTMDB'
}


def main():
    one_day = time.strftime("%Y%m%d", time.localtime(time.time() - 246060))
    opts, args = getopt.getopt(sys.argv[1:], 'd:')
    for op, value in opts:
        if op == '-d':
            m = re.search('[0-9]{8}', value)
            if m:
                one_day = value
            else:
                print("请输入8位日期(比如：20130215)")
                return 'no'
        print("正在统计会员鲜花库存(%s)…" % one_day)

    db_conn = MySQLdb.connect(user=optmap['dbuser'],
                              passwd=optmap['dbpass'],
                              host=optmap['dbhost'],
                              port=optmap['dbport'],
                              db=optmap['dbname'])
    db_conn.query("use %s" % optmap['dbname'])
    db_cursor = db_conn.cursor()

    vip_user_list = {}
    for i in range(10):
        sql = "select USERID, VIPSTATE from VIPUSER%s" % i
        print(sql)
        db_cursor.execute(sql)
        db_rows = db_cursor.fetchall()
        for USERID, VIPSTATE in db_rows:
            vip_user_list[USERID] = VIPSTATE

    vip_user_flower_list = {}
    for i in range(10):
        sql = "select USERID, FLOWER from VIPUSERFLOWER%s" % i
        print(sql)
        db_cursor.execute(sql)
        db_rows = db_cursor.fetchall()
        for USERID, FLOWER in db_rows:
            vip_user_flower_list[USERID] = FLOWER

    vip_state_flower_list = {}
    vip_state_flower_list[1] = 0
    vip_state_flower_list[2] = 0
    vip_state_flower_list[3] = 0
    for key in vip_user_list:
        if key in vip_user_flower_list:
            if vip_user_list[key] in vip_state_flower_list:
                vip_state_flower_list[
                    vip_user_list[key]] += vip_user_flower_list[key]

    for key in vip_state_flower_list:
        print(key, vip_state_flower_list[key])

    db_cursor.close()
    db_conn.close()

    db_conn = MySQLdb.connect(user=optmap['dbuser'],
                              passwd=optmap['dbpass'],
                              host=optmap['dbhost_gm'],
                              port=optmap['dbport'],
                              db=optmap['dbname_gm'])
    db_conn.query("use %s" % optmap['dbname_gm'])
    db_cursor = db_conn.cursor()

    dword_time = time.mktime(time.strptime(one_day, '%Y%m%d'))

    sql = "update VIPUSERFLOWERMONTHLY set year_flower_left_num=%d, month_flower_left_num=%d, week_flower_left_num=%d where count_time='%d'" % (
        vip_state_flower_list[3], vip_state_flower_list[2],
        vip_state_flower_list[1], dword_time)
    print(sql)
    db_conn.query(sql)
    db_conn.commit()

    db_cursor.close()
    db_conn.close()


# if name == "main":
main()

# -----------------------------------
# !/usr/bin/env python
# --coding:utf-8--
# 会员信息统计
import MySQLdb
import os, sys, re, string
import time, getopt

optmap = {
    'dbuser': 'haoren',
    'dbpass': 'ddddd',
    'dbhost': '172.17.1.13',
    'dbport': 3306,
    'dbname': 'GTMDB',
    'logdir': '/home/haoren/logdir/',  # 外网环境日志目录
    'logpattern': '^sessionserver.log.'  # 外网环境日志名称前缀
}


def get_files(dir, pattern):
    print(dir, pattern)
    match_file_list = []
    if os.path.exists(dir):
        cur_file_list = os.listdir(dir)
        for file_name in cur_file_list:
            if re.search(pattern, file_name):
                match_file_list.append(file_name)
                return match_file_list
            else:
                return 'no'


def main():
    one_day = time.strftime("%Y%m%d", time.localtime(time.time() -
                                                     246060))  # 默认日期为脚本运行的上一天
    opts, args = getopt.getopt(sys.argv[1:], 'd:')
    for op, value in opts:
        if op == '-d':
            m = re.search('[0-9]{8}', value)
            if m:
                one_day = value
        else:
            print("请输入8位日期(比如：20130215)")
            return 'no'

    print("正在读取VIP用户数据(%s)..." % one_day)
    db_conn = MySQLdb.connect(user=optmap['dbuser'],
                              passwd=optmap['dbpass'],
                              host=optmap['dbhost'],
                              port=optmap['dbport'],
                              db=optmap['dbname'])
    db_cursor = db_conn.cursor()

    temp_vip_active_user_num_file_name = '/tmp/vipactiveusernumtemp.txt'
    command = "cat /dev/null > %s" % (temp_vip_active_user_num_file_name)
    os.system(command)

    if re.search('haoren', optmap['logdir']):
        print('外网环境')
        log_dir_name_list = get_files(optmap['logdir'], one_day[2:])
        for log_dir_name_item in log_dir_name_list:
            log_dir_full_path = optmap['logdir'] + log_dir_name_item + '/'
            log_file_name_list = get_files(log_dir_full_path,
                                           optmap['logpattern'] + one_day[2:])
            for log_file_name_item in log_file_name_list:
                print(log_file_name_item)
                command = "cat %s%s |awk '/用户登录/' |awk '/vip状态/' >> %s" % (
                    log_dir_full_path, log_file_name_item,
                    temp_vip_active_user_num_file_name)
                os.system(command)
    else:
        print('内网环境')
        log_file_name_list = get_files(optmap['logdir'],
                                       optmap['logpattern'] + one_day[2:])
        for log_file_name_item in log_file_name_list:
            command = "cat %s%s |awk '/用户登录/' |awk '/vip状态/' >> %s" % (
                optmap['logdir'], log_file_name_item,
                temp_vip_active_user_num_file_name)
            os.system(command)

    command = "cat %s |wc -l" % temp_vip_active_user_num_file_name
    os.system(command)

    # 一天当中用户可能从月会员降级到周会员，造成不同会员状态的同一帐号统计两次，所以总会员!=年会员+月会员+周会员)
    # 不同状态的会员用同一计算机登录，所以总mac/ip!=年mac/ip+月mac/ip+周mac/ip
    total_account_map = {}
    total_mac_map = {}
    total_ip_map = {}
    before_account_map = {}
    before_mac_map = {}
    before_ip_map = {}

    account_map = {1: {}, 2: {}, 3: {}, 11: {}, 12: {}, 13: {}}
    mac_map = {1: {}, 2: {}, 3: {}, 11: {}, 12: {}, 13: {}}
    ip_map = {1: {}, 2: {}, 3: {}, 11: {}, 12: {}, 13: {}}

    temp_vip_active_user_num_file = open(temp_vip_active_user_num_file_name)
    for one_line in temp_vip_active_user_num_file.readlines():
        match = re.search(
            "^(\S+) SS\[\d+\] TRACE: 用户登录:imid:(\d+),mac地址:(\d+),ip地址:(\d+),vip状态:(\d+),登录时间:(\d+)(\S+)",
            one_line)
        if match:
            if string.atoi(match.group(5)) in (1, 2, 3):
                total_account_map[string.atoi(match.group(2))] = string.atoi(
                    match.group(5))
                total_mac_map[string.atoi(match.group(3))] = string.atoi(
                    match.group(5))
                total_ip_map[string.atoi(match.group(4))] = string.atoi(
                    match.group(5))
            elif string.atoi(match.group(5)) in (11, 12, 13):
                before_account_map[string.atoi(match.group(2))] = string.atoi(
                    match.group(5))
                before_mac_map[string.atoi(match.group(3))] = string.atoi(
                    match.group(5))
                before_ip_map[string.atoi(match.group(4))] = string.atoi(
                    match.group(5))
            account_map[string.atoi(match.group(5))][string.atoi(
                match.group(2))] = string.atoi(match.group(3))
            mac_map[string.atoi(match.group(5))][string.atoi(
                match.group(3))] = string.atoi(match.group(2))
            ip_map[string.atoi(match.group(5))][string.atoi(
                match.group(4))] = string.atoi(match.group(2))
    temp_vip_active_user_num_file.close()

    dword_time = time.mktime(time.strptime(one_day, '%Y%m%d'))
    db_conn.query("use %s" % optmap['dbname'])
    sql = "delete from VIPACTIVEUSERNUM where active_time='%d'" % dword_time
    print(sql)
    db_conn.query(sql)

    sql = "insert into VIPACTIVEUSERNUM (active_time) values('%d')" % (
        dword_time)
    print(sql)
    db_conn.query(sql)

    sql = "update VIPACTIVEUSERNUM set year_account_num=%d, year_mac_num=%d, year_ip_num=%d, month_account_num=%d, month_mac_num=%d, month_ip_num=%d, week_account_num=%d, week_mac_num=%d, week_ip_num=%d, total_mac_num=%d, total_ip_num=%d, before_account_num=%d, before_mac_num=%d, before_ip_num=%d where active_time='%d'" % (
        len(account_map[3]), len(mac_map[3]), len(ip_map[3]),
        len(account_map[2]), len(mac_map[2]), len(
            ip_map[2]), len(account_map[1]), len(mac_map[1]), len(ip_map[1]),
        len(total_mac_map), len(total_ip_map), len(before_account_map),
        len(before_mac_map), len(before_ip_map), dword_time)
    print(sql)
    db_conn.query(sql)
    db_conn.commit()

    db_cursor.close()
    db_conn.close()


# if name == "main"
main()
