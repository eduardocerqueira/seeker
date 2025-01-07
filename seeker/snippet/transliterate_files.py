#date: 2025-01-07T16:44:31Z
#url: https://api.github.com/gists/898209e1df126fa1da4d209e17715e11
#owner: https://api.github.com/users/Tholv777

import os
from transliterate import translit

def transliterate_filename(filename):
    """Transliterate a filename from Cyrillic to Latin if needed."""
    base, ext = os.path.splitext(filename)
    if ext.lower() == ".mp3" and any("а" <= char <= "я" or "А" <= char <= "Я" for char in base):
        return translit(base, 'ru', reversed=True) + ext
    return filename

def rename_files_in_folder(folder_path):
    """Rename all MP3 files with Cyrillic names in the specified folder."""
    for file in os.listdir(folder_path):
        original_path = os.path.join(folder_path, file)
        if os.path.isfile(original_path):
            new_name = transliterate_filename(file)
            if new_name != file:
                new_path = os.path.join(folder_path, new_name)
                os.rename(original_path, new_path)
                print(f"Renamed: {file} -> {new_name}")

if __name__ == "__main__":
    folder_path = os.getcwd()  # Get current working directory
    print(f"Processing folder: {folder_path}")
    rename_files_in_folder(folder_path)
    print("Renaming complete!")
