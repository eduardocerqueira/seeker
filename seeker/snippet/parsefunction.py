#date: 2023-04-10T16:57:47Z
#url: https://api.github.com/gists/93857c703da18b63b30956d0ef03dfed
#owner: https://api.github.com/users/hericlibong


def parse(self, response):
        script_content = response.xpath('//script[contains(., "dates")]/text()').extract_first()
        date_list = json.loads(script_content)
        date_list = date_list['props']['pageProps']['pageData']['ranking']['dates']
        for item_id in date_list:
            url = f"https://www.fifa.com/api/ranking-overview?locale=en&dateId={item_id['id']}"
            date_text = item_id['text'].replace('Sept', 'sep') # Set the date to the correct format
            date_obj = datetime.strptime(date_text, '%d %b %Y')
            date = date_obj.strftime('%Y-%m-%d')
            yield scrapy.Request(url=url, callback=self.parse_ranking_data, meta={'url': url,'date': date})
          