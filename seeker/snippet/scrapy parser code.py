#date: 2021-09-03T16:56:53Z
#url: https://api.github.com/gists/73d796105402064c19cd341604bc518f
#owner: https://api.github.com/users/vaitybharati

import scrapy

class AmazonReviewSpider(scrapy.Spider):
    name = 'amazon_review'
    allowed_domains = ['amazon.in']
    start_urls = ['http://amazon.in']

    def parse(self, response):
        pass