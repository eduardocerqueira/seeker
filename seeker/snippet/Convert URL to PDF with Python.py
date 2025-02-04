#date: 2025-02-04T17:02:23Z
#url: https://api.github.com/gists/2dfe7334a23d544b01b9d601a30e7574
#owner: https://api.github.com/users/aspose-com-kb

# Import necessary modules
import requests  # For making HTTP requests to fetch webpage content
from io import BytesIO  # To handle byte stream data
from aspose.pdf import Document  # Import Aspose PDF's Document class for PDF operations
import aspose.pdf as ap  # Import Aspose PDF module for additional functionality

def fetch_web_content_as_stream(webpage_url):
    """
    Fetches the content of a webpage and returns it as a byte stream.
    
    Parameters:
        webpage_url (str): The URL of the webpage to fetch.
    
    Returns:
        BytesIO: A byte stream of the webpage content.
    """
    response = requests.get(webpage_url)  # Send GET request to the specified URL
    response.raise_for_status()  # Raise an error if the request fails
    return BytesIO(response.content)  # Return the content as a byte stream

def main():
    """
    Main function that converts a webpage into a PDF document.
    """

    # Set Aspose.PDF license (assumes "license.lic" file is available)
    license = ap.License()
    license.set_license("license.lic")

    # Define the webpage URL to be converted
    webpage_url = "https://docs.aspose.com/"

    # Configure HTML-to-PDF conversion options
    pdf_options = ap.HtmlLoadOptions(webpage_url)  # Create HTML load options with the webpage URL
    pdf_options.page_info.width = 1200  # Set PDF page width
    pdf_options.page_info.height = 850  # Set PDF page height

    # Fetch webpage content as a byte stream
    with fetch_web_content_as_stream(webpage_url) as web_stream:

        # Uncomment the lines below to print and inspect the webpage content
        # print(web_stream.read().decode('utf-8', errors='ignore'))
        # web_stream.seek(0)  # Reset the stream position after reading

        # Create a PDF document from the webpage stream
        pdf_document = Document(web_stream, pdf_options)
        
        # Save the converted PDF document
        pdf_document.save("Converted_WebPage.pdf")

    print("URL converted to PDF successfully")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
