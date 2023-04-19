#date: 2023-04-19T16:42:31Z
#url: https://api.github.com/gists/443a4153885a7385d64406949439d965
#owner: https://api.github.com/users/rfeers

# Void lists to store the data.
job_title_list = [];
job_company_list = [];
job_location_list = [];
job_salary_list = [];
job_type_list = [];
job_date_list = [];
job_description_list = [];
job_link_list = [];
job_id_list = [];

# The next button is defined.
next_button_xpath = '//*[@id="jobsearch-JapanPage"]/div/div/div[5]/div[1]/nav/div[6]/a'


num_jobs_scraped = 0

while num_jobs_scraped < 1000:
    
    # The job browing is started
    job_page = driver.find_element(By.ID,"mosaic-jobResults")
    jobs = job_page.find_elements(By.CLASS_NAME,"job_seen_beacon") # return a list
    num_jobs_scraped = num_jobs_scraped + len(jobs)
    
    
    for ii in jobs: 
        # Finding the job title and its related elements
        job_title = ii.find_element(By.CLASS_NAME,"jobTitle")
        job_title_list.append(job_title.text)
        job_link_list.append(job_title.find_element(By.CSS_SELECTOR,"a").get_attribute("href"))
        job_id_list.append(job_title.find_element(By.CSS_SELECTOR,"a").get_attribute("id"))
        
        # Finding the company name and location
        job_company_list.append(ii.find_element(By.CLASS_NAME,"companyName").text)
        job_location_list.append(ii.find_element(By.CLASS_NAME,"companyLocation").text)
        # Finding the posting date
        job_date_list.append(ii.find_element(By.CLASS_NAME,"date").text)
        
        # Trying to find the salary element. If it is not found, a None will be returned. 
        try: 
            job_salary_list.append(ii.find_element(By.CLASS_NAME,"salary-snippet-container").text)
            
        except: 
            try: 
                job_salary_list.append(ii.find_element(By.CLASS_NAME,"estimated-salary").text)
            except: 
                job_salary_list.append(None)

        # We wait a random amount of seconds to imitate a human behavior. 
        time.sleep(random.random())
        
        #Click the job element to get the description
        job_title.click()
        
        #Wait for a bit for the website to charge (again with a random behavior)
        time.sleep(1+random.random())
        
        #Find the job description. If the element is not found, a None will be returned.
        try: 
            job_description_list.append(driver.find_element(By.ID,"jobDescriptionText").text)
            
        except: 
            job_description_list.append(None)
    
    time.sleep(1+random.random())
    
    # We press the next button. 
    driver.find_element(By.XPATH,next_button_xpath).click()
    
    
    # The button element is updated to the 7th button instead of the 6th.
    next_button_xpath = '//*[@id="jobsearch-JapanPage"]/div/div/div[5]/div[1]/nav/div[7]/a'