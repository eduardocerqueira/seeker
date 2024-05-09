#date: 2024-05-09T16:51:06Z
#url: https://api.github.com/gists/0334ccce0757786ebcf04119922b27e1
#owner: https://api.github.com/users/joegoldin

"""Main module."""

import logging
import urllib.request
import xml.etree.ElementTree as ET
import datetime


def get_url_xml(url: str) -> ET.Element:
    "Send a GET request to the given URL and parse the response as an XML object"
    response = urllib.request.urlopen(url).read()
    # logging.info("Response: %s", response)
    return ET.fromstring(response)


# def get_sitemaps_from_xml(sitemap: ET.Element) -> list[str]:
#     sitemaps: list[str] = []
#     for sitemap in sitemaps:
#         if "Sitemap" in str(object=site.tag):
#            logging.info(get_sitemaps_from_xml(sitemap))
#     return sitemaps


def extract_sitemaps(sitemap: ET.Element, start_date: datetime.datetime = None) -> list[str]:
    url: str = ""
    date: datetime.datetime = None
    for child in sitemap:
        if "loc" in str(child.tag):
            url = child.text
        if "lastmod" in str(child.tag):
            lastmod = child.text
            date = datetime.datetime.fromisoformat(
                lastmod).replace(tzinfo=datetime.timezone.utc)

    if url and (start_date is not None and date is not None and date > start_date):
        return url

    return None



def sitemap_urls(sitemap: ET.Element, start_date: datetime.datetime) -> list[str]:
    urls: dict[str, int] = {}
    for site in sitemap:
        logging.info(site.tag)
        if "sitemap" in str(site.tag):
            sitemap_url = extract_sitemaps(site, start_date)
            logging.info("Extracted: %s", sitemap_url)
            # if sitemap not in sitemaps:
            #     sitemaps.append(sitemap)
            
            actual_url: str = ""
            sitemap_xml = get_url_xml(sitemap_url)
            date: datetime.datetime = None
            for child_node in sitemap_xml:
                for child in child_node:
                    logging.info("child: %s", str(child.tag))
                    if "loc" in str(child.tag):
                        actual_url = child.text
                    if "lastmod" in str(child.tag):
                        lastmod = child.text
                        date = datetime.datetime.fromisoformat(
                            lastmod).replace(tzinfo=datetime.timezone.utc)
                if actual_url and (start_date is not None and date is not None and date > start_date):
                    urls[actual_url] = 1
                
    logging.info("urls: %s", urls)
    return urls


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] [%(levelname)s] - %(message)s")

    final_urls = sitemap_urls(get_url_xml(
        "https://gist.githubusercontent.com/nikitawootten/198e9b7a235abfc1165fb53d64397416/raw/e09a6a9a0719d619b879a785162430ff25fb9fbd/recursive_sitemap.xml"),
    start_date = datetime.datetime.fromisoformat("2022-11-19T09:51:40.000-05:00").replace(tzinfo=datetime.timezone.utc)).keys()
    logging.info(list(final_urls))
