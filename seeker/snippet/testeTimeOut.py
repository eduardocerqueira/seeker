#date: 2022-07-08T16:56:48Z
#url: https://api.github.com/gists/12093da599b8c5c96720752c2dae962d
#owner: https://api.github.com/users/Wanhenri

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

options = Options()
options.headless = False

browser = webdriver.Firefox(options=options, executable_path=r'/mnt/c/Users/wanderson.santos/Documents/Projeto2w/geckodriver.exe')

pagina = False
cont = 1
while pagina == False:
    browser.get('https://www.ana.gov.br/www/AcoesAdministrativas/CDOC/consultaExter.asp')
    
    delay = 3 # seconds

    if cont == 9:
        browser.get('https://www.ana.gov.br/www/AcoesAdministrativas/CDOC/consultaExterna.asp')

    try:
        myElem = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.ID, 'IdOfMyElement')))
        print ("Page is ready!")
        pagina = True
        browser.close()
        continue
    except TimeoutException:
        if cont == 10:
            pagina = True
            browser.close()
        print(cont)
        print ("Loading took too much time!")
        cont+=1
        continue