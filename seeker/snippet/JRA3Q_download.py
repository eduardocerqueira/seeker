#date: 2023-08-16T17:05:55Z
#url: https://api.github.com/gists/103f51e7a2c25cc27aef5f2bc227936f
#owner: https://api.github.com/users/nicholas512

#!/usr/bin/env python

import urllib
import urllib.request
import urllib.parse
from http.cookiejar import MozillaCookieJar
from html.parser import HTMLParser
import subprocess
import sys
import os


import optparse
import netrc
import getpass


class CASLoginParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.action = None
        self.data = {}

    def handle_starttag(self, tagname, attribute):
        if tagname.lower() == 'form':
            attribute = dict(attribute)
            if 'action' in attribute:
                self.action = attribute['action']
        elif tagname.lower() == 'input':
            attribute = dict(attribute)
            if 'name' in attribute and 'value' in attribute:
                self.data[attribute['name']] = attribute['value']


class DIASAccess():
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
        self.__cas_url = 'https://auth.diasjp.net/cas/login?'
        self.__username = username
        self.__password = "**********"
        self.__cj = MozillaCookieJar()
        self.__opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.__cj))

    def open(self, url, data=None):
        response = self.__opener.open(url, data)
        response_url = response.geturl()

        if response_url != url and response_url.startswith(self.__cas_url):
            # redirected to CAS login page
            response = self.__login_cas(response)
            if data is not None:
                # If POST (data != None), need reopen
                response.close()
                response = self.__opener.open(url, data)

        return response

    def __login_cas(self, response):
        parser = CASLoginParser()
        parser.feed(response.read().decode('utf-8'))
        parser.close()

        if parser.action is None:
            raise LoginError('Not login page')

        action_url = urllib.parse.urljoin(response.geturl(), parser.action)
        data = parser.data
        data['username'] = self.__username
        data['password'] = "**********"

        response.close()
        response = self.__opener.open(action_url, urllib.parse.urlencode(data).encode("utf-8"))

        if response.geturl() == action_url:
            print('Authorization fail')
            quit()

        return response

    def dl(self, url, path, file, data=None):
        try:
            response = self.__opener.open(url, data)
            if not os.path.exists('.' + path):
                os.makedirs('.' + path)

            f = open('.' + path + file, 'wb')
            file_size_dl = 0
            block_size = 8192
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)

            f.close
            print(path + file + "  OK")
            return response

        except urllib.request.HTTPError:
            print(path + file + "  NG")


class LoginError(Exception):
    def __init__(self, e):
        Exception.__init__(self, e)

if __name__ == '__main__':
    host = 'data.diasjp.net'

    usage ='''usage: %prog [options]'''
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-n', '--netrc', default=None,
                      help='specify the netrc file', metavar='FILE')
    parser.add_option('-u', '--user', default=None,
                      help='specify the DIAS account name',
                      metavar='USERNAME')

    (options, args) = parser.parse_args()

    (login, password) = "**********"

    try:
      auth = netrc.netrc(options.netrc).authenticators(host)
      if auth is not None:
            (login, account, password) = "**********"
    except (IOError):
      pass

    if options.user is not None:
        login = options.user
        password = "**********"

    if login is None:
        login = input('Username: ')

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"i "**********"s "**********"  "**********"N "**********"o "**********"n "**********"e "**********": "**********"
        password = getpass.getpass('Password: "**********"

    access = "**********"


    targeturl='https://data.diasjp.net/dl/storages/filelist/dataset:645'
    response = access.open(targeturl)
    response.close()


    access.dl('https://data.diasjp.net/dl/storages/downloadCmd/L0pSQTNRL0hpc3QvRGFpbHkvYW5sX3AxMjUvMTk5NzEwL2FubF9wMTI1X2RlcHIuMTk5NzEwLmN0bA==', '/JRA3Q/Hist/Daily/anl_p125/199710/', 'anl_p125_depr.199710.ctl')

    access.dl('https://data.diasjp.net/dl/storages/downloadCmd/L0pSQTNRL0hpc3QvRGFpbHkvYW5sX3AxMjUvMTk5NzEwL2FubF9wMTI1X2RlcHIuMTk5NzEwLmlkeA==', '/JRA3Q/Hist/Daily/anl_p125/199710/', 'anl_p125_depr.199710.idx')

 