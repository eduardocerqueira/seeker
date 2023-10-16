#date: 2023-10-16T17:06:07Z
#url: https://api.github.com/gists/c826a1f1b4de77f11437478f28a98f63
#owner: https://api.github.com/users/greenmojo2

import requests
from bs4 import BeautifulSoup
import re
import os

urlToScrape = input("Enter the URL to scrape: ")
# if URL is not empty, then use it. Otherwise, use default URL
if urlToScrape != "":
    url = urlToScrape
else:
    url = "https://bobs-burgers.fandom.com/wiki/Bob%27s_Burgers_Wiki"

SpecifiedFolder = input("Enter the folder name: ")
# if specified folder is not empty, then use it. Otherwise, use default folder
if SpecifiedFolder != "":
    folder = SpecifiedFolder
else:
    folder = "Bob's Burgers Wiki"
# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

RecursionDepth = input("Enter the recursion depth: ")
# if recursion depth is not empty, then use it. Otherwise, use default depth
if RecursionDepth != "":
    RecursionDepth = int(RecursionDepth)
else:
    RecursionDepth = 3

# List of URL patterns to exclude from scraping
exclude_patterns = [
    r"Special:",
    r"action=edit",
    r"WikiCommunity Portal",
]

# List to collect URLs of pages without main content
pages_without_content = []

# List to keep track of scraped URLs
scraped_urls = []

# Function to clean a string and make it a valid filename
def clean_filename(text):
    # Remove characters that are not valid in filenames
    text = re.sub(r'[\/:*?"<>|]', '', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    return text

# Function to get the title of a page
def get_title(soup, url):
    # Attempt to find a title from header elements
    for header_level in range(1, 7):
        header = soup.find(f"h{header_level}")
        if header:
            return clean_filename(header.get_text())
    
    # If no header is found, try using a portion of the URL path
    path_segments = url.split("/")
    if len(path_segments) >= 2:
        potential_title = path_segments[-1].replace("_", " ")
        return clean_filename(potential_title)
    
    # If all else fails, return "Unknown Title"
    return "Unknown Title"

# Function to scrape a page and its subpages recursively
def scrape_page_recursive(url, output_folder, exclude_patterns, depth=0, max_depth=RecursionDepth):
    if depth > max_depth:
        return

    # Skip scraping if the URL matches any of the excluded patterns
    if any(re.search(pattern, url) for pattern in exclude_patterns):
        print(f"Skipping page: {url} (matches exclusion pattern)")
        return

    # Skip scraping if the URL has already been scraped
    if url in scraped_urls:
        print(f"Skipping page: {url} (already scraped)")
        return

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Add the URL to the list of scraped URLs
        scraped_urls.append(url)

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Check if there is a main content section
        main_content = soup.find("div", {"class": "mw-parser-output"})
        if not main_content:
            pages_without_content.append(url)
            print(f"Page without main content: {url}")
            return  # Skip further processing for this page
        
        # Find and remove the table cell containing the "Behind the scenes" section
        behind_the_scenes_cell = soup.find("th", style="background: #FF9933;")
        if behind_the_scenes_cell:
            behind_the_scenes_cell.find_parent("table").decompose()

        # Find and remove the table of contents
        toc = soup.find("div", {"class": "toc"})
        if toc:
            toc.extract()

        # Find and remove reference links ([1], [2], etc.) and the associated references section
        for reference in soup.find_all("sup", {"class": "reference"}):
            reference.extract()
        references_section = soup.find("div", {"class": "reflist"})
        if references_section:
            references_section.extract()

        # Find and remove the "External links" section and its contents
        external_links_section = soup.find("span", {"id": "External_links"})
        if external_links_section:
            next_tag = external_links_section.find_next()
            while next_tag and next_tag.name != "h2":
                next_tag.extract()
                next_tag = external_links_section.find_next()

        # Remove the unwanted sections
        unwanted_sections = [
            "First",
            "Gallery",
            "Main article: Bob Belcher/Gallery",
            "Archer Version",
            "Main article: Archer/Bob's Burgers connection"
        ]

        for section_name in unwanted_sections:
            section = soup.find("span", text=section_name)
            if section:
                section.extract()

        # Extract the main content from the page
        main_content = soup.find("div", {"class": "mw-parser-output"})

        # Extract the text from the main content and remove empty square brackets
        text = main_content.get_text()
        text = re.sub(r'\[\s*\]', '', text)

        # Create a unique filename based on the guessed title
        title = get_title(soup, url)
        filename = f"{title}.txt"

        # Specify 'utf-8' encoding when writing to the new file
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(text)

        print(f"Content from {url} has been saved to '{output_path}'.")
        
        # Find and scrape links on the page
        links = soup.find_all("a", href=True)
        for link in links:
            subpage_url = link['href']
            if subpage_url.startswith("/wiki/"):
                # Construct the full URL by appending the subpage URL to the base URL
                full_subpage_url = f"https://bobs-burgers.fandom.com{subpage_url}"
                # Call the scrape_page_recursive function to scrape the subpage
                scrape_page_recursive(full_subpage_url, output_folder, exclude_patterns, depth=depth + 1)

    else:
        print(f"Failed to retrieve the page {url}. Status code:", response.status_code)

# Starting URL
main_url = url
# main_url = "https://bobs-burgers.fandom.com/wiki/Bob%27s_Burgers_Wiki"

# Call the scrape_page_recursive function to start the recursive scraping
scrape_page_recursive(main_url, folder, exclude_patterns)

# Print the list of pages without main content
print("Pages without main content:")
for page_url in pages_without_content:
    print(page_url)
