#date: 2024-06-28T16:22:56Z
#url: https://api.github.com/gists/705f36e1a9953b9d9ede9764c57ed293
#owner: https://api.github.com/users/rix4uni

import argparse
from bs4 import BeautifulSoup
import requests

def scrape_firebounty(output_file):
    # Get the total number of pages dynamically
    initial_url = "https://firebounty.com/?page=1&sort=created_at&search_field=name&order=desc&search="
    initial_response = requests.get(initial_url)
    initial_soup = BeautifulSoup(initial_response.content, "html.parser")

    # Find the second-to-last li element within the pagination
    pagination = initial_soup.find("ul", class_="pagination pagination-sm")
    second_last_li = pagination.find_all("li")[-2]
    total_pages = int(second_last_li.text.strip())

    # Open the output file for appending
    with open(output_file, 'a') as file:
        # Loop through each page up to the total number of pages
        for i in range(1, total_pages + 1):
            url = "https://firebounty.com/?page={}&sort=created_at&search_field=name&order=desc&search=".format(i)

            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            divs = soup.find_all("div", {"class": "Rtable-row", "data-url": True})
            urls = [div["data-url"] for div in divs]

            for url in urls:
                b = "https://firebounty.com" + url

                response = requests.get(b)
                soup = BeautifulSoup(response.content, "html.parser")

                # Find the first element with the class 'buttonColor'
                first_button_element = soup.find(class_='buttonColor')

                if first_button_element:
                    # Extract the href attribute
                    href = first_button_element.get('href')
                    if href:
                        print(href)
                        # Write the href to the output file
                        file.write(href + '\n')
                        file.flush()  # Flush the buffer to ensure immediate write

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape Firebounty Programs List')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file to save the hrefs')
    
    args = parser.parse_args()
    
    scrape_firebounty(args.output)
