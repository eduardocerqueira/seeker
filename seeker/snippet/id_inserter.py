#date: 2024-10-02T17:05:18Z
#url: https://api.github.com/gists/b401e4d681c2eb4ed7d051738c0b7e23
#owner: https://api.github.com/users/Pymmdrza

from bs4 import BeautifulSoup
import sys
import os

def generate_id_from_text(text):
    """
    Generates a valid id from text by converting it to lower case,
    removing special characters, and replacing spaces with hyphens.
    """
    return text.lower().replace(" ", "-").replace(",", "").replace(".", "").replace("'", "").replace("\"", "").replace("&", "and")

def create_table_of_contents(soup, headers):
    """
    Create a table of contents (TOC) for the given headers and return it as a list of HTML tags.
    """
    toc = soup.new_tag("ul")  # Main TOC list

    # Iterate through each header and create links to them
    for header in headers:
        # Create <li><a href="#header-id">Header Text</a></li>
        toc_item = soup.new_tag("li")
        link = soup.new_tag("a", href=f"#{header['id']}")
        link.string = header.get_text()
        toc_item.append(link)
        toc.append(toc_item)

    return toc

def process_html_file(file_path):
    """
    Process the given HTML file, add IDs to headers if missing, and create a Table of Contents (TOC).
    Saves the modified file with '_new_edition' suffix in the same directory.
    """
    # Read the content of the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')

    # Find all header tags (h1 to h5) and check if they have ids, if not, generate ids
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])
    for header in headers:
        if not header.get('id'):  # If id is not present
            header['id'] = generate_id_from_text(header.get_text())

    # Create the Table of Contents
    toc = create_table_of_contents(soup, headers)

    # Insert the TOC at the beginning of the body
    if soup.body:
        soup.body.insert(0, toc)  # Insert TOC at the beginning of <body>
    else:
        soup.insert(0, toc)  # Insert TOC at the beginning of the document if <body> not found

    # Generate the new filename with '_new_edition' suffix
    base_name, ext = os.path.splitext(file_path)
    new_file_path = f"{base_name}_new_edition{ext}"

    # Save the modified content to a new file
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))

    print(f"Successfully processed and saved the modified file as: {new_file_path}")

# Main entry point for script execution
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python id_inserter.py <HTML_FILE_PATH>")
    else:
        html_file = sys.argv[1]
        if not os.path.isfile(html_file):
            print(f"Error: File '{html_file}' not found.")
        else:
            process_html_file(html_file)
