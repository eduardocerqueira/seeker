#date: 2025-02-25T16:48:28Z
#url: https://api.github.com/gists/d99c0d51a22a5ce40dfa4973a3046776
#owner: https://api.github.com/users/dduyg

import os
import shutil
import logging

# Define the file categories and their associated extensions
categories = {
    'Images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
    'Documents': ['.pdf', '.doc', '.docx', '.txt', '.ppt', '.pptx', '.xls', '.xlsx'],
    'Audio': ['.mp3', '.wav', '.aac', '.flac'],
    'Videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
    'Archives': ['.zip', '.tar', '.gz', '.rar'],
    'Code': ['.py', '.html', '.css', '.js', '.java', '.cpp', '.php'],
    'Others': []
}

# Set up logging configuration
logging.basicConfig(filename='file_organizer.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def organize_files(directory):
    # Check if the provided directory exists
    if not os.path.exists(directory):
        logging.error(f"Error: The directory {directory} does not exist.")
        print(f"Error: The directory {directory} does not exist.")
        return

    # Loop through the categories and create subfolders if they don't exist
    for category in categories:
        category_path = os.path.join(directory, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)
            logging.info(f"Created folder: {category_path}")

    # Walk through all files and subdirectories in the given directory
    for root, dirs, files in os.walk(directory):
        # Skip the base directory since it's already being handled
        if root == directory:
            continue

        # Move files into the correct folders
        for filename in files:
            filepath = os.path.join(root, filename)

            # Determine the file's extension
            file_extension = os.path.splitext(filename)[1].lower()

            # Find the corresponding category for the file
            moved = False
            for category, extensions in categories.items():
                if file_extension in extensions:
                    category_path = os.path.join(directory, category)
                    shutil.move(filepath, os.path.join(category_path, filename))
                    logging.info(f"Moved {filename} from {root} to {category} folder.")
                    moved = True
                    break

            # If no category matched, move it to 'Others'
            if not moved:
                others_path = os.path.join(directory, 'Others')
                shutil.move(filepath, os.path.join(others_path, filename))
                logging.info(f"Moved {filename} from {root} to Others folder.")

    # Handle the base directory files as well
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            file_extension = os.path.splitext(filename)[1].lower()
            moved = False
            for category, extensions in categories.items():
                if file_extension in extensions:
                    category_path = os.path.join(directory, category)
                    shutil.move(filepath, os.path.join(category_path, filename))
                    logging.info(f"Moved {filename} from {directory} to {category} folder.")
                    moved = True
                    break

            if not moved:
                others_path = os.path.join(directory, 'Others')
                shutil.move(filepath, os.path.join(others_path, filename))
                logging.info(f"Moved {filename} from {directory} to Others folder.")

if __name__ == "__main__":
    # Input directory to organize
    directory = input("Enter the path of the directory you want to organize: ").strip()

    organize_files(directory)
    print("File organization complete!")
    logging.info("File organization complete!")