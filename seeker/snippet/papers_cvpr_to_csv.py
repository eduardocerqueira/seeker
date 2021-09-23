#date: 2021-09-23T17:04:24Z
#url: https://api.github.com/gists/1e3af390e48d2d6abe3ebced940ea5fb
#owner: https://api.github.com/users/luutp

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
papers_cvpr_to_csv.py

Description:

This script uses Selenium to extract information from CVPR conference url and Arxiv and saves to .csv file. Only papers that are available on Arxiv are included.

The paper info are: ["Title","Authors","Abstract","Citation","Date","Arxiv_url", "Pdf_url","Ppwcode_url","Notes"]

Selenium requires ChromeDriver which can be downloaded at:
https://chromedriver.chromium.org/downloads

Author: PhatLuu
Contact: tpluu2207@gmail.com
Created on: 2021/09/23
"""
#%%
# =================================== IMPORT PACKAGES ===============================

# Standard Packages
import inspect
import os
import re

# Data Analytics
import pandas as pd

# Web pages and URL Related
from selenium import webdriver
from selenium.webdriver.support import ui

# GUI and User Interfaces Packages
import pyautogui
import pyperclip

# Utilities
import time
from tqdm import tqdm


# =====================================DEFINE=========================================

script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.dirname(script_dir)

# =====================================START===========================================


def get_active_url():
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.2)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(0.2)
    return pyperclip.paste()


class webDriver(object):
    # Initialize Class
    def __init__(self, driver, wait):
        self._driver = driver
        self._wait = wait

    def get_xpath(self, xpath, attr=None):
        driver = self._driver
        wait = self._wait
        output_str = ""
        try:
            el = wait.until(lambda driver: driver.find_element_by_xpath(xpath))
            time.sleep(1)
            if attr is None:
                output_str = el.text
            else:
                output_str = el.get_attribute(attr)
        except Exception as e:
            print(e)
        return output_str

    def driver_click(self, xpath):
        driver = self._driver
        wait = self._wait
        el = wait.until(lambda driver: driver.find_element_by_xpath(xpath))
        time.sleep(2)
        el.click()

    def get_hrefs(self, xpath=None):
        driver = self._driver
        wait = self._wait
        if xpath is not None:
            element = wait.until(lambda driver: driver.find_element_by_xpath(xpath))
            time.sleep(1)
        else:
            element = driver
        hrefs = [
            (x.text, x.get_attribute("href"))
            for x in element.find_elements_by_tag_name("a")
        ]
        return hrefs

    def __str__(self):
        str_output = ""
        for mem in self.get_obj_attrs(self):
            str_output += f"{mem[0]:25s}: {mem[1]}\n"
        return str_output

    @staticmethod
    def get_obj_attrs(obj):
        members = inspect.getmembers(obj)
        attr_members = []
        for mem in members:
            if not inspect.isfunction(mem[1]) and not inspect.ismethod(mem[1]):
                if not mem[0].startswith("__"):
                    attr_members.append(mem)
        return attr_members


class arxivClass(webDriver):
    # Initialize Class
    def __init__(self, driver, wait):
        self._driver = driver
        self._wait = wait
        self.title = self.get_title()
        self.authors = self.get_authors()
        self.date = self.get_date()
        self.abstract = self.get_abstract()
        self.url = self.get_url()
        self.note = self.get_note()
        self.category = self.get_category()
        self.citation = self.get_citation()
        self.community_code = self.get_community_code()

    def get_title(self):
        xpath = "//h1[@class='title mathjax']"
        return self.get_xpath(xpath)

    def get_authors(self):
        xpath = "//div[@class='authors']"
        r = self.get_xpath(xpath)
        author_list = r.split(",")
        return author_list

    def get_date(self):
        xpath = "//div[@class='dateline']"
        r = self.get_xpath(xpath)
        pattern = r"\d{1,2} [A-z]{3} \d{4}"
        output_dates = re.findall(pattern, r)
        return output_dates[-1]

    def get_abstract(self):
        xpath = "//blockquote[@class='abstract mathjax']"
        r = self.get_xpath(xpath)
        return r

    def get_url(self):
        xpath = "//a[@class='abs-button download-pdf']"
        pdf_link = self.get_xpath(xpath, attr="href")
        return pdf_link

    def get_community_code(self):
        hrefs = []
        try:
            xpath = "//div[@class='labstabs']/label[2]"
            self.driver_click(xpath)
            xpath = "//div[@id='pwc-output']"
            all_hrefs = self.get_hrefs(xpath)
            for href in all_hrefs:
                if len(href[0]):
                    hrefs.append(href)
            return hrefs
        except Exception as e:
            print(e)
        return hrefs

    def get_citation(self):
        xpath = "//div[@id='labstabs']/div[@class='labstabs']/label[1]"
        self.driver_click(xpath)
        xpath = "//div[@class='column lab-switch']/label[@class='switch']/span[@class='slider']"
        self.driver_click(xpath)
        xpath = "//div[@id='col-citations']/div[@class='bib-col-header']/span[@class='bib-col-center']/a[@class='bib-col-title']"
        r = self.get_xpath(xpath)
        return r

    def get_note(self):
        xpath = "//div[@class='metatable']/table/tbody/tr[1]/td[@class='tablecell comments mathjax']"
        r = self.get_xpath(xpath)
        return r

    def get_category(self):
        xpath = (
            "//div[@class='extra-services']/div[@class='browse']/div[@class='current']"
        )
        r = self.get_xpath(xpath)
        return r

    @property
    def first_author(self):
        return self.authors[0]

    @property
    def last_author(self):
        return self.authors[-1]

    @property
    def year(self):
        return self.date[-4:]

    @property
    def id(self):
        return os.path.basename(self.url)

    @property
    def paperwithcode_url(self):
        url = None
        if len(self.community_code):
            for href in self.community_code:
                if "paperswithcode" in href[1].lower():
                    url = href[1]
        return url

    @property
    def github_url(self):
        url = None
        if len(self.community_code):
            for href in self.community_code:
                if "github" in href[0].lower():
                    url = href[1]
        return url

    @property
    def bibtex(self):
        """BibTex string of the reference."""
        author = self.first_author.split(" ")[-1]
        year = self.year
        short_title = " ".join(self.title.split(" ")[:3])
        newbib_id = self.drop_chars(f"{author}{year}{short_title}", [".", "-", " "])
        newbib_id = self.get_valid_filename(newbib_id.lower())
        lines = ["@article{" + newbib_id]
        for k, v in [
            ("Author", " and ".join(self.authors)),
            ("Title", self.title),
            ("Eprint", self.id),
            ("ArchivePrefix", "arXiv"),
            ("PrimaryClass", self.category),
            ("Year", self.year),
            ("Note", self.note),
            ("Url", self.url),
        ]:
            if len(v):
                lines.append("%-13s = {%s}" % (k, v))

        return ("," + os.linesep).join(lines) + os.linesep + "}"

    # ========================================================================

    @staticmethod
    def get_valid_filename(s):
        s = s.strip().replace(" ", "_")
        return re.sub(r"(?u)[^-\w.]", "", s)

    @staticmethod
    def drop_chars(s, chars):
        output = s
        for char in chars:
            output = output.replace(char, "")
        return output


def get_valid_filename(input_filename):
    """Summary:
	--------
	Return valid filename by removing empty space and special characters

	Inputs:
	-------
		input_filename (str): input filename

	Returns:
	--------
		str: valid filename
	"""

    input_filename = input_filename.strip().replace(" ", "_")
    return re.sub(r"(?u)[^-\w.]", "_", input_filename)


# ==================================== MAIN ========================================


def main(url=None, output_dir=None):
    """Summary:
    --------
    This script uses Selenium to extract information from CVPR conference url and Arxiv and saves to .csv file. Only papers that are available on Arxiv are included.

    The paper info are:
                ["Title",
                "Authors",
                "Abstract",
                "Citation",
                "Date",
                "Arxiv_url",
                "Pdf_url",
                "Ppwcode_url",
                "Notes"]
    
    Selenium requires ChromeDriver which can be downloaded at:
    https://chromedriver.chromium.org/downloads

    Inputs:
    -------
        url ([str], optional): [description]. url to CVPR conference. For example https://openaccess.thecvf.com/CVPR2020?day=2020-06-16. Defaults to None.
        output_dir ([str], optional): [description]. Directory to save .csv file. Defaults to None.
    """
    # Local Path to Chrome Driver
    DRIVER_PATH = os.path.join(os.path.expanduser("~"), "_chromedriver/chromedriver")
    # Optional.
    LEFT_SCREEN = (200, 50)

    # Get url link currently open on Left Screen if no url is provided
    if url is None:
        pyautogui.click(LEFT_SCREEN)
        time.sleep(0.2)
        url = get_active_url()
    print(f"Processing for: {url}")
    # Define output dir for csv file
    if output_dir is None:
        output_dir = os.path.join(project_dir, "docs/papers")
        csv_filename = get_valid_filename(os.path.basename(url)) + ".csv"
        csv_filepath = os.path.join(output_dir, csv_filename)

    driver = webdriver.Chrome(executable_path=DRIVER_PATH)
    wait = ui.WebDriverWait(driver, 10)
    driver.get(url)
    xpath = "//div[@id='content']/dl/dd"
    els = wait.until(lambda driver: driver.find_elements_by_xpath(xpath))
    arXiv_list = []
    for el in tqdm(els):
        for x in el.find_elements_by_tag_name("a"):
            href = x.get_attribute("href")
            if href is not None and "arxiv" in href:
                arXiv_list.append(href)
                break

    for arxiv_url in tqdm(arXiv_list):
        print(f"Process for :{arxiv_url}")
        driver = webdriver.Chrome(executable_path=DRIVER_PATH)
        wait = ui.WebDriverWait(driver, 10)
        driver.get(arxiv_url)
        article = arxivClass(driver, wait)
        title = article.title
        authors = ", ".join(article.authors)
        abstract = article.abstract
        raw_citation = article.citation
        pattern = r"\((.*?)\)"
        citation = (
            re.search(pattern, raw_citation).group(1)
            if re.search(pattern, raw_citation) is not None
            else 0
        )
        date = article.date
        pdf_url = article.url
        arxiv_url = pdf_url.replace("pdf", "abs")
        ppwcode_url = article.paperwithcode_url
        note = article.note

        data = [
            title,
            authors,
            abstract,
            citation,
            date,
            arxiv_url,
            pdf_url,
            ppwcode_url,
            note,
        ]
        driver.close()

        df = pd.DataFrame(
            columns=[
                "Title",
                "Authors",
                "Abstract",
                "Citation",
                "Date",
                "Arxiv_url",
                "Pdf_url",
                "Ppwcode_url",
                "Notes",
            ],
            data=[data],
        )
        if not os.path.isfile(csv_filepath):
            print(f"Create new csv file: {csv_filepath}")
            df.to_csv(csv_filepath, index=False)
        else:
            print(f"Append to current csv file: {csv_filepath}")
            df.to_csv(csv_filepath, mode="a", index=False, header=False)

    print("DONE")


if __name__ == "__main__":
    main()