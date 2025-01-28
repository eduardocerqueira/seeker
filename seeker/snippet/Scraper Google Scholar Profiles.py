#date: 2025-01-28T17:03:34Z
#url: https://api.github.com/gists/c70149985f59745a744d179058dc886f
#owner: https://api.github.com/users/SofiaECalle

import pandas as pd
from time import sleep
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.maximize_window()

#driver = webdriver.Firefox()
#driver.maximize_window()
import random


driver.get("https://scholar.google.es/citations?view_op=search_authors&hl=es&mauthors=universidad+nacional+de+educaci%C3%B3n+UNAE&before_author=C25-_wIBAAAJ&astart=0")
sleep(4)

profile_list=[]


for i in range(1,15):
    sleep(2)
    profiles = driver.find_elements(By.XPATH,'//h3[@class="gs_ai_name"]/a')
    for p in profiles:
        profile = p.get_attribute('href')
        profile_list.append(profile)
    
    try:
        driver.find_element(By.XPATH,'//button[@type="button"][2]').click()
    except:
        break
'''
profiles = driver.find_elements(By.XPATH,'//h3[@class="gs_ai_name"]/a')   
for p in profiles:
    profile = p.get_attribute('href')
    profile_list.append(profile)
'''
data = []
for pr in profile_list:
    driver.get(pr)
        
        
    name=driver.find_element(By.XPATH,'//div[@id="gsc_prf_in"]').text
    verify=driver.find_element(By.XPATH,'(//div[@class="gsc_prf_il"])[2]').text
    Citas_Total=driver.find_element(By.XPATH,'(//td[@class="gsc_rsb_std"])[1]').text
    Citas_Desde=driver.find_element(By.XPATH,'(//td[@class="gsc_rsb_std"])[2]').text
    
    Indice_h_Total=driver.find_element(By.XPATH,'(//td[@class="gsc_rsb_std"])[3]').text
    Indice_h_Desde=driver.find_element(By.XPATH,'(//td[@class="gsc_rsb_std"])[4]').text
    
    Indice_i10_Total=driver.find_element(By.XPATH,'(//td[@class="gsc_rsb_std"])[3]').text
    Indice_i10_Desde=driver.find_element(By.XPATH,'(//td[@class="gsc_rsb_std"])[4]').text
    
    for i in range(1,7):
        sleep(2)
        button=driver.find_element(By.XPATH,'//button[@class="gs_btnPD gs_in_ib gs_btn_flat gs_btn_lrge gs_btn_lsu"]').click()
        
    aritcals= driver.find_elements(By.XPATH,'//tr[@class="gsc_a_tr"]')
    for a in aritcals:
        article_link = a.find_element(By.XPATH,'.//a[@class="gsc_a_at"]').get_attribute('href')
        article_title = a.find_element(By.XPATH,'.//a[@class="gsc_a_at"]').text
        citado= a.find_element(By.XPATH,'.//td[@class="gsc_a_c"]/a').text
        anoo= a.find_element(By.XPATH,'.//td[@class="gsc_a_y"]/span').text
        data.append([name, verify, Citas_Total, Citas_Desde, Indice_h_Total, Indice_h_Desde,
                    Indice_i10_Total, Indice_i10_Desde, article_title, article_link, citado, anoo])

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=['Name', 'Verify', 'Citas_Total', 'Citas_Desde', 'Indice_h_Total',
                                'Indice_h_Desde', 'Indice_i10_Total', 'Indice_i10_Desde', 'Article_Title',
                                'Article_Link', 'Citado', 'Anoo'])

# Save DataFrame to an Excel sheet
df.to_excel(f'profile_dataall.xlsx', index=False)
driver.quit()