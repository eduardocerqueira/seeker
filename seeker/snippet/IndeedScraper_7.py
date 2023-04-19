#date: 2023-04-19T16:43:50Z
#url: https://api.github.com/gists/53ff8d0839f350413a776f53f2441575
#owner: https://api.github.com/users/rfeers

country_list = [country_name]*len(job_title_list)
job_name_list = [job_name]*len(job_title_list)
scraped_date_list = [date.today()]*len(job_title_list)

indeed_job_data = pd.DataFrame({
    'job_id': job_id_list,
    'scraped_date': scraped_date_list,
    'country': country_list,
    'job_name': job_name_list,
    'job_post_date': job_date_list,
    'job_company': job_company_list,
    'job_title': job_title_list,
    'job_location': job_location_list,
    'job_description': job_description_list,
    'job_link': job_link_list,
    'job_salary': job_salary_list,
    
})

indeed_job_data.to_csv("Output/{0}_{1}_".format(country_name,job_name)+"_ddbb.csv")