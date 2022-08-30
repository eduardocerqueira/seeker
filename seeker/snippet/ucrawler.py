#date: 2022-08-30T16:54:29Z
#url: https://api.github.com/gists/bd961e422d9478fdee6fdc97b1ae2242
#owner: https://api.github.com/users/AgnesDigitan

import requests, re
from bs4 import BeautifulSoup
import tqdm
 
headers = {
    "Connection" : "keep-alive",
    "Cache-Control" : "max-age=0",
    "sec-ch-ua-mobile" : "?0",
    "DNT" : "1",
    "Upgrade-Insecure-Requests" : "1",
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
    "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Sec-Fetch-Site" : "none",
    "Sec-Fetch-Mode" : "navigate",
    "Sec-Fetch-User" : "?1",
    "Sec-Fetch-Dest" : "document",
    "Accept-Encoding" : "gzip, deflate, br",
    "Accept-Language" : "ko-KR,ko;q=0.9"
    }

if __name__ == "__main__":
    outstream = open("output.txt","w",encoding="utf-8")
    errstream = open("err.txt","w",encoding="utf-8")
    needcheckstream = open("need_check.txt","w",encoding="utf-8")
    scores = []
    last_url_poi = ""
    for i in tqdm.tqdm(range(100),total=100):
        url = f"https://gall.dcinside.com/mgallery/board/lists/?id=umamusu&sort_type=N&search_head=160&page={i}"
        res = requests.get(url, headers=headers)
        if res.url == last_url_poi:break
        last_url_poi = res.url
        soup = BeautifulSoup(res.text, "lxml")
        article_list = soup.select(".us-post")
        for element in article_list:
            num = element.select(".gall_num")[0].text
            title = element.select(".ub-word > a")[0].text
            
            regex = re.compile(r'(\d{1,12}((,|.)\d{3})*(\.\d+)?)')
            search = regex.search(title)
            if not search:
                errstream.write(f"Not found\t{num}\t{title}\n")
                continue
            score = search.group(1)
            try:
                score = score.replace("원","")
                if "만" in score:
                    score = score.split("만")
                    score = int(score[0]) * 10000 + int(score[1])
                else:
                    score = int(score.replace(",", "").replace(".","").replace(" ","").strip())
                if score < 3000:
                    if title[search.span()[1]] == "만": score = score * 10000
                    else:
                        needcheckstream.write(f"{num}\t{title}\t{score}\n")
                        continue
                outstream.write(f"{num}\t{title}\t{score}\n")
                scores.append(score)
            except:
                errstream.write(f"Not int\t{num}\t{title}\n")
    print(sum(scores))
    print(sum(scores) / len(scores))
