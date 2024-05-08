#date: 2024-05-08T17:00:11Z
#url: https://api.github.com/gists/f77345353b1fb5c5013ba72e20688d07
#owner: https://api.github.com/users/nhtranngoc

import requests

# Universal header, pretend we're a browser yay
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
book_prefixes = ["Pre1", "Prefs1", "Key1", "Kfs1", "ELT_First4", "ELT_FIRST5", "ELT_Adv4"]
test_count = 4

# PET has 4 sections
# KET has 5 sections
# FCE has 4 sections
# CAE has 4 sections
# Generate a list of names for each audio section in each test in each book. (yea)
def generate_urls():
    urls = []
    for prefix in book_prefixes:
        for test in range(1, test_count + 1):
            section_count = 4
            if (prefix == "Key1") or (prefix == "Kfs1"):
                section_count = 5

            for section in range (1, section_count + 1):
                urls.append(prefix + "_t" + str(test) + "_audio" + str(section))

    return urls    

if __name__ == "__main__":
    urls = generate_urls()

    for url in urls:
        print("REQUESTING: File " + url)

        doc = requests.get('http://cambridge.org/' + url, allow_redirects=True, headers=headers)

        if (doc.status_code != 200):
            print("REQUEST FAILED: File " + url + " unable to download, status code " + doc.status_code)
            continue
        
        with open(url + '.mp3', 'wb') as f:
            f.write(doc.content)

        print("REQUESTED: File " + url + " successfully")
        print("----")