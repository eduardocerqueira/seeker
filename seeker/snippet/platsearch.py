#date: 2021-11-24T16:55:03Z
#url: https://api.github.com/gists/86f07f2d3a743273a8c32fe132eb8cd3
#owner: https://api.github.com/users/plasmaphase

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from multiprocessing import Process
from multiprocessing import Pool
import time

plat_majors = [6, 85, 10, 55, 2, 50, 9, 75, 7, 65, 30, 25, 20, 4, 40, 1, 58, 5, 8]

def platSearch(parcel_major):

    df = pd.DataFrame(columns=['Plat_ID'])
    df.head()

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--enable-javascript')
    driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver', options=chrome_options)
    filename = 'PlatIDs_' + str("{0:02}".format(parcel_major)) + '.csv'
    
    for minor in range(0, 999):
        platid_str = str("{0:05}".format((parcel_major * 1000) + minor))
        url = 'https://gis.co.carver.mn.us/platsearch/'
        driver.get(url)
        content = driver.page_source
        scripttxt = 'performPlatNumSearch(\"' + platid_str + '\",\"\")'
        driver.execute_script(scripttxt)
        time.sleep(1)
        try:
            elem = driver.find_element_by_id('platSpan_1')
        except NoSuchElementException:
            print(platid_str + " not found")
        else:
            print(platid_str + " found!")
            for option in elem.find_elements_by_tag_name('option'):
                df = df.append({'Plat_ID': option.get_attribute("value")}, ignore_index=True)
            
       

    df.to_csv(filename, index=False, encoding='utf-8')
    driver.close()

if __name__ == "__main__":
    
    task = []
    for maj in plat_majors:
        p = Process(target=platSearch, args=(maj,))
        task.append(p)        
        p.start()
        