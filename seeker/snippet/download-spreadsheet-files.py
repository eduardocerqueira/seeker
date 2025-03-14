#date: 2025-03-14T16:45:14Z
#url: https://api.github.com/gists/3334afa699b2c0fa2f8e34d4296652a9
#owner: https://api.github.com/users/jordangarrison

import argparse
import pandas as pd
import requests
import os
from urllib.parse import urlparse

def download_files_from_excel(excel_file, output_dir):
    """
    Reads an Excel file, downloads files from URLs in the 'URL' column,
    and saves them with filenames from the 'filename' column into a specified directory.

    Args:
        excel_file (str): Path to the Excel file.
        output_dir (str): Path to the output directory.
    """

    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        print(f"Error: Excel file not found at {excel_file}")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    if 'URL' not in df.columns or 'filename' not in df.columns:
        print("Error: Excel file must contain 'URL' and 'filename' columns.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        url = row['URL']
        filename = row['filename']

        if pd.isna(url) or pd.isna(filename):
            print(f"Skipping row {index + 2} due to missing URL or filename.")  # Excel rows are 1-indexed
            continue

        filepath = os.path.join(output_dir, filename)

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded: {filename} from {url} to {filepath}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename} from {url}: {e}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from URLs listed in an Excel file.")
    parser.add_argument("excel_file", help="Path to the Excel file.")
    parser.add_argument("output_dir", help="Path to the output directory.")

    args = parser.parse_args()

    download_files_from_excel(args.excel_file, args.output_dir)