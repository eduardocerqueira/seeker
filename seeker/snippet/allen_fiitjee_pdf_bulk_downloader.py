#date: 2024-10-02T16:42:22Z
#url: https://api.github.com/gists/ad59585e3a8c4bdedc8e65b1b0044151
#owner: https://api.github.com/users/datavorous

import requests
import os
import logging

# Setup logging
logging.basicConfig(
    filename="file_downloader.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Paths to store results
OUTPUT_FILE_FIITJEE = "positive_urls_fiitjee.txt"
OUTPUT_FILE_allen = "positive_urls_allen.txt"

# Directories to store downloads
DOWNLOAD_DIR_FIITJEE = "downloaded_pdfs_fiitjee"
DOWNLOAD_DIR_allen = "downloaded_pdfs_allen"
os.makedirs(DOWNLOAD_DIR_FIITJEE, exist_ok=True)
os.makedirs(DOWNLOAD_DIR_allen, exist_ok=True)

# ------------------- FIITJEE Downloader -------------------

# FIITJEE Base URL
BASE_URL_FIITJEE = "https://cms.fiitjee.com/Resources/DownloadCentre/Document_Pdf_{ID}.pdf"

# Function to check URL status and ensure it is a PDF for FIITJEE
def check_url_status_fiitjee(file_id):
    url = BASE_URL_FIITJEE.format(ID=file_id)
    try:
        response = requests.head(url)
        status_code = response.status_code
        if status_code == 200 and response.headers.get("Content-Type") == "application/pdf":
            with open(OUTPUT_FILE_FIITJEE, "a") as file:
                file.write(f"{url}\n")
            logging.info(f"Valid PDF found (FIITJEE): {url}")
            return True
        else:
            logging.warning(f"Invalid or non-PDF URL (FIITJEE): {url} (Status: {status_code})")
            return False
    except Exception as e:
        logging.error(f"Error checking (FIITJEE) {url}: {str(e)}")
        return False

# ------------------- allen Downloader -------------------

# allen Base URL
BASE_URL_allen = "https://d2yn992461j5av.cloudfront.net/Dat{YEAR}Live/document/test/{TYPE1}/{TYPE2}_Report_{ID}.pdf"
YEARS = [2022, 2023, 2024]
TYPE1_VALUES = ["question", "solution"]
TYPE2_VALUES = ["Question", "Solution"]

# Function to check URL status and save valid PDFs for allen
def check_url_status_allen(year, type1, type2, file_id):
    url = BASE_URL_allen.format(YEAR=year, TYPE1=type1, TYPE2=type2, ID=file_id)
    try:
        response = requests.head(url)
        status_code = response.status_code
        if status_code == 200 and response.headers.get("Content-Type") == "application/pdf":
            with open(OUTPUT_FILE_allen, "a") as file:
                file.write(f"{url}\n")
            logging.info(f"Valid PDF found (allen): {url}")
            return True
        else:
            logging.warning(f"Invalid or non-PDF URL (allen): {url} (Status: {status_code})")
            return False
    except Exception as e:
        logging.error(f"Error checking (allen) {url}: {str(e)}")
        return False

# ------------------- Download Functionality -------------------

# Function to optionally download files from a file of URLs
def download_files_from_file(file_path, download_dir):
    try:
        with open(file_path, "r") as file:
            urls = file.readlines()
        for url in urls:
            url = url.strip()
            file_name = url.split("/")[-1]
            download_file(url, file_name, download_dir)
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")

# Function to download a single file
def download_file(url, file_name, download_dir):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(download_dir, file_name)
            with open(file_path, "wb") as file:
                file.write(response.content)
            logging.info(f"Downloaded: {file_name}")
        else:
            logging.warning(f"Failed to download: {file_name} (Status Code: {response.status_code})")
    except Exception as e:
        logging.error(f"Error downloading {file_name}: {str(e)}")

# ------------------- Brute Force Functions -------------------

# Brute-force FIITJEE IDs
def brute_force_fiitjee(start_id=1, end_id=100):
    for file_id in range(start_id, end_id + 1):
        check_url_status_fiitjee(file_id)

# Brute-force allen IDs
def brute_force_allen(start_id=1, end_id=100):
    for year in YEARS:
        for type1, type2 in zip(TYPE1_VALUES, TYPE2_VALUES):
            for file_id in range(start_id, end_id + 1):
                check_url_status_allen(year, type1, type2, file_id)

# ------------------- Main Execution -------------------

if __name__ == "__main__":
    # Brute-force FIITJEE IDs from 1 to 100 (you can adjust the range)
    brute_force_fiitjee(1, 100)

    # Brute-force allen IDs from 1 to 100 (you can adjust the range)
    brute_force_allen(1, 100)

    # Uncomment below to download files from stored URLs
    # download_files_from_file(OUTPUT_FILE_FIITJEE, DOWNLOAD_DIR_FIITJEE)
    # download_files_from_file(OUTPUT_FILE_allen, DOWNLOAD_DIR_allen)
