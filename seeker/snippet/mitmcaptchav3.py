#date: 2022-09-09T17:17:48Z
#url: https://api.github.com/gists/1985d7c086e8ef38f7ddde1034246b3f
#owner: https://api.github.com/users/Fr0st3h

#Created By Fr0st3h#7019

from mitmproxy.net.http.http1.assemble import assemble_request
from mitmproxy import http, ctx
import pathlib
import requests
import random
import pathlib

proxy = {'http': f'http://geo.iproyal.com:22323', 'https': f'http://geo.iproyal.com:22323'}
reloadFromSolve = False
proxies = list()

def load(loader):
    loader.add_option(
        name="proxies",
        typespec=str,
        default="",
        help="NoNeed",
    )

def request(flow):
    global proxies
    global reloadFromSolve
    if(len(proxies) == 0):
        proxies = ctx.options.proxies.split(",")
    if(flow.request.url == "https://blizzard-api.arkoselabs.com/fc/gt2/public_key/E8A75615-1CBA-5DFF-8032-D16BCF234E10"):
        headers = {
            'Host': 'blizzard-api.arkoselabs.com',
            'Sec-Ch-Ua': '"Chromium";v="103", ".Not/A)Brand";v="99"',
            'Sec-Ch-Ua-Mobile': '?0',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Accept': '*/*',
            'Origin': 'https://blizzard-api.arkoselabs.com',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Dest': 'empty',
            'Referer': 'https://blizzard-api.arkoselabs.com/v2/E8A75615-1CBA-5DFF-8032-D16BCF234E10/enforcement.8b144c4f9762e265309b45a275be4e56.html',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        newCap = requests.post('https://blizzard-api.arkoselabs.com/fc/gt2/public_key/E8A75615-1CBA-5DFF-8032-D16BCF234E10', headers=headers, data=flow.request.text, proxies=proxy)
        token = "**********"
        reqData = "**********"=canvas&sid=eu-west-1&lang=&analytics_tier=40&token={token}&data[status]=init'
        captchaInfo = requests.post('https://blizzard-api.arkoselabs.com/fc/gfct/', headers=flow.request.headers, data=reqData)
        foundGoodCaptcha = 0
        if(len(captchaInfo.json()["game_data"]["customGUI"]["_challenge_imgs"]) > 1):
            while foundGoodCaptcha < 5:
                newCap = requests.post('https://blizzard-api.arkoselabs.com/fc/gt2/public_key/E8A75615-1CBA-5DFF-8032-D16BCF234E10', headers=headers, data=flow.request.text, proxies=proxy)
                token = "**********"
                reqData = "**********"=canvas&sid=eu-west-1&lang=&analytics_tier=40&token={token}&data[status]=init'
                captchaInfo = requests.post('https://blizzard-api.arkoselabs.com/fc/gfct/', headers=flow.request.headers, data=reqData)
                if(len(captchaInfo.json()["game_data"]["customGUI"]["_challenge_imgs"]) <= 1):
                    flow.response = http.Response.make(#Return 0 click captcha
                        200,
                        newCap.content,
                        {"Content-Type": "application/json"},
                    )
                    return
                else:
                    foundGoodCaptcha += 1
        flow.response = http.Response.make(#Failed to find 0 click captcha, return original captcha
            200,
            newCap.content,
            {"Content-Type": "application/json"},
        )
    
    if(flow.request.url == 'https://account.battle.net/creation/flow/creation-full/step/get-started'):
        flow.request.host = 'account.battle.net'
        flow.request.url = 'https://account.battle.net/creation/flow/creation-full'
        flow.request.method = "GET"#Changes request so it GETs the main page again and allows you to solve another captcha
        postData = assemble_request(flow.request).decode('utf-8')
        token = "**********"="arkose"\r\n\r\n')[1]
        token = "**********"
        print(token)
        reloadFromSolve = True
        
        
def response(flow):
    global reloadFromSolve
    global proxies
            
    if(flow.request.url == 'https://account.battle.net/creation/flow/creation-full'):
        if(b"Too many attempts" in flow.response.content):
            randomProxy = random.choice(proxies).strip('\n')
            getProxy = {'http': f'http://{randomProxy}', 'https': f'http://{randomProxy}'}
            try:
                newPage = requests.get('https://account.battle.net/creation/flow/creation-full', headers=flow.response.headers, proxies=getProxy, timeout=7)
                flow.response = http.Response.make(
                    200,
                    newPage.content,
                    {"Content-Type": "text/html"},
                )
            except:
                pass
            
        if(flow.request.url == 'https://account.battle.net/creation/flow/creation-full'):
            textToReplace = [[b"Get Started", b"Hexogen Captcha v3"], 
                            [b"Let's verify some information about you to help set up your account.", b""], 
                            [b"Learn why we need this", b"Please click continue"], 
                            [b"The following information helps us verify your identity and provide you with appropriate content and settings.", b"For Hexogen to make an account, please click the continue button."]]
            for index in textToReplace:
                flow.response.content = flow.response.content.replace(index[0], index[1])

            if(reloadFromSolve):
                flow.response.content = flow.response.content.replace(b"class=\"battlenet-logo\"", b"")
                reloadFromSolve = Falsent.replace(b"class=\"battlenet-logo\"", b"")
                reloadFromSolve = False