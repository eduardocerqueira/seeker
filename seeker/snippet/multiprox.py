#date: 2024-02-29T17:10:28Z
#url: https://api.github.com/gists/3498267e0a017181da0e50501f2c86b5
#owner: https://api.github.com/users/m-tq

    def startThreads(self):
        proxyList = []
        proxySSH = ''
        path = ''
        Url = ''
        try:
            conf = json.loads(open('config.json','r').read())
            proxySSH = conf['proxySSH']
            Url = conf['ytLink']
        except:
            print('[!] Error opening config file.')
            
        if proxySSH == 'True' :
            print('[*] Using SSH Proxy Tunnel..')
            print('[*] For first time running SSH Proxy Tunnel, please click "Accept" on popup..')
            proxyList = self._startSSH()
            print('[*] Please wait until all proxy ready.. arround ~30 sec..')
            time.sleep(30)
        else :
            print('[*] Using Custom Proxy..')
            proxies = open('proxylist.txt','r').readlines()
            proxyList = [proxy.replace('\n','') for proxy in proxies]             
        useragent = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246', 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.5195.52 Safari/537.36', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/604.4.7 (KHTML, like Gecko) Version/11.0.2 Safari/604.4.7','Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:101.0) Gecko/20100101 Firefox/101.0','Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:99.0) Gecko/20100101 Firefox/99.0']
        count = 30
        rang = 10
        position = [[i,i] for i in range(0,count*rang,rang)]
        threads = []
        for i in range(count):
            threads.append(Thread(target=self._getDriver, args=[ position[i], Url, webdriver.Chrome]))
        for thread in threads:
            thread.start()
            time.sleep(3)
        for thread in threads:
            thread.join()
        