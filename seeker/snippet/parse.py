#date: 2022-05-18T16:53:07Z
#url: https://api.github.com/gists/29536e1c2d9039fa0eab4129ca0e4786
#owner: https://api.github.com/users/srang992

def parse(self, response, **kwargs):
    job_desc = response.css('div.show-more-less-html__markup ::text').extract()
    criterion = response.css('span.description__job-criteria-text::text').extract()
    title = response.css('h1.topcard__title::text').extract()
    company = response.css('a.topcard__org-name-link::text').extract()
    loc = response.css('span.topcard__flavor::text').extract()
    time = response.css('span.posted-time-ago__text::text').extract()

    clean_desc = " ".join(job_desc).strip().replace(",", "|")
    clean_criterion = " ".join(crit.strip() for crit in criterion).strip().replace(",", "|")
    clean_title = " ".join(title).strip()
    clean_company = " ".join(company).strip()
    clean_loc = " ".join(loc).strip().replace(",", "|")
    clean_time = " ".join(time).strip()

    all_items = {'title': clean_title, 'company': clean_company, 'description': clean_desc,
                 'criterion': clean_criterion, 'location': clean_loc, 'time_posted': clean_time}

    yield all_items