#date: 2023-02-10T16:45:29Z
#url: https://api.github.com/gists/1615cc1b8e7691b1d651117cd3a88e0e
#owner: https://api.github.com/users/JesperDramsch

# Author: Jacob Rust
# Modified: Jesper Dramsch
# Date: 7/8/2013
# Description:This script organizes downloaded files into separate folders depending
# on the file type. This was made for windows systems.
# Download directory should be the first command line argument
# New file types can be added to the dictionary in the main method

# Usage: python organize.py
import os
import sys
import shutil
import hashlib
from pathlib import Path
from tqdm import tqdm
from typing import Union, Tuple, List

strpath = Union[str, os.PathLike]


def safe_move(src_path: strpath, dst_path: strpath) -> None:
    """Safely moves folder and file to new location

    These assume full paths including the filename, so you have control over the name.

    Parameters
    ----------
    src_path : str or Path
        Location of item to be moved
    dst_path : str or Path
        Destination of the item to be moved
    """
    try:
        os.rename(src_path, dst_path)
    except OSError:
        print("Hard moving disks")
        shutil.move(src_path, dst_path)


def move_folders(download_directory: strpath, download_folders: List[str], filetypes: dict) -> None:
    """Move the folders in the download directory

    Parameters
    ----------
    download_directory : strpath
        Root directory
    download_folders : List[str]
        Folders detected in the root direcoty
    filetypes : dict
        Filetypes we have a mapping for in main()
    """
    print("Moving folders...")
    for item in tqdm(download_folders):
        if item not in filetypes.values():
            src_path = Path(download_directory, item)
            dst_path = Path(download_directory, "Folders", item)
            safe_move(src_path, dst_path)


def move_files(download_directory: strpath, download_files: List[str], filetypes: dict) -> None:
    """Move all the files in the download folder

    Parameters
    ----------
    download_directory : strpath
        Root directory
    download_files : List[str]
        List of files detected in the root directory
    filetypes : dict
        Filetypes we have a mapping for in main()
    """
    print("Moving files...")
    add_keys = set()
    for filename in tqdm(download_files):
        out = move_file(filename, download_directory, filetypes)
        if out:
            add_keys.add(out)
    return add_keys


def categorize_files_folders(download_directory: strpath, filetypes: dict) -> Tuple[List[str], List[str]]:
    """Categorise items in root into files and folders

    Filters folders by target folders

    Parameters
    ----------
    download_directory : str or Path
        Root directory
    filetypes : dict
        Filetypes we have a mapping for in main()

    Returns
    -------
    list, list
        Lists with files and folders separated
    """
    download_files, download_folders = [], []
    for item in os.listdir(download_directory):
        if item in filetypes.values():
            continue
        elif Path(download_directory, item).is_dir():
            download_folders.append(item)
        else:
            download_files.append(item)
    return download_files, download_folders


def create_folders(download_directory: strpath, filetypes: dict) -> None:
    """Create Folders in filetypes defined in main()

    Parameters
    ----------
    download_directory : str or Path
        Root directory
    filetypes : dict
        Filetypes we have a mapping for in main()
    """
    for filetype in filetypes.values():
        directory = Path(download_directory, filetype)
        directory.mkdir(parents=True, exist_ok=True)


# Moves file to its proper folder and delete any duplicates
def move_file(move_file: str, download_directory: strpath, filetypes: dict) -> Union[str, None]:
    """Move file

    Checks if it exists at target and modifies name if different

    Parameters
    ----------
    move_file : str
        Name of file to move
    download_directory : str or Path
        Root directory
    filetypes : dict
        Filetypes we have a mapping for in main()

    Returns
    -------
    Union[str, None]
        None if successful, otherwise returns the suffix
    """
    move_file = Path(move_file)

    # Ignore directories
    if move_file.is_dir():
        print(move_file)
        return

    # Get suffix to check against suffix
    suffix = move_file.suffix.lower()[1:]
    if suffix in filetypes.keys():
        src_path = Path(download_directory, move_file)
        dst_path = Path(download_directory, filetypes[suffix], move_file)

        # If the file doesn't have a duplicate in the new folder, move it
        if not dst_path.is_file():
            safe_move(src_path, dst_path.with_suffix(dst_path.suffix))
        # If the file already exists with that name and has the same md5 sum
        elif dst_path.is_file() and (checksum(src_path) == checksum(dst_path)):
            os.remove(src_path)
            print("removed " + str(src_path))
        else:
            safe_move(
                src_path,
                dst_path.with_name(dst_path.stem + "-" + str(checksum(dst_path))[:6]).with_suffix(dst_path.suffix),
            )
    else:
        return suffix


# Get md5 checksum of a file. Chunk size is how much of the file to read at a time.
def checksum(filedir: strpath, chunksize: int = 8192) -> str:
    """Generate checksum of file

    Parameters
    ----------
    filedir : str or Path
        File to check
    chunksize : int, optional
        size of chunks to calculate, by default 8192

    Returns
    -------
    str
        Returns the checksum of file
    """
    md5 = hashlib.md5()
    with open(filedir, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            # If the chunk is empty, reached end of file so stop
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def main(download_directory: str) -> None:
    """Define filetypes and execute main logic with moves

    Parameters
    ----------
    download_directory : str
        Root dir
    """
    # Dictionary contains file types as keys and lists of their corresponding file formats
    filetypes = {}
    filetypes.update(
        {
            key: "Images"
            for key in [
                "ai",
                "avif",
                "bmp",
                "drawio",
                "drx",
                "eps",
                "gif",
                "jfif",
                "jpeg",
                "jpg",
                "png",
                "ppm",
                "psd",
                "svg",
                "tiff",
                "webp",
                "xcf",
            ]
        }
    )
    filetypes.update({key: "Audio" for key in ["aac", "aiff", "flac", "mp3", "ogg", "wav"]})
    filetypes.update(
        {
            key: "Video"
            for key in ["flv", "m3u8", "m4a", "m4v", "mkv", "mov", "mp4", "mpe", "mpeg", "mpg", "webm", "wmv"]
        }
    )
    filetypes.update(
        {
            key: "Documents"
            for key in ["bib", "doc", "docx", "ini", "md", "odt", "pdf", "ppt", "pptx", "rst", "rtf", "tex", "txt"]
        }
    )
    filetypes.update({key: "Executables" for key in ["exe", "msi"]})
    filetypes.update({key: "Archives" for key in ["7", "7z", "gz", "rar", "tar", "zip"]})
    filetypes.update({key: "Apps" for key in ["apk", "img", "iso", "ova", "vmdk"]})
    filetypes.update({key: "Spreadsheets" for key in ["csv", "json", "xml", "xls", "xlsx"]})
    filetypes.update({key: "Web" for key in ["html", "tmpl"]})
    filetypes.update({key: "Fonts" for key in ["otf", "ttf"]})
    filetypes.update({key: "Office" for key in ["ics", "eml", "mbox", "potx"]})
    filetypes.update({key: "Books" for key in ["acsm", "epub", "mobi"]})
    filetypes.update({key: "Code" for key in ["css", "ipynb", "ipynb_files", "js", "php", "py", "yaml"]})
    filetypes.update({key: "Backup" for key in ["bak", "backup", "old", "save"]})

    # Single file suffixes
    filetypes["apkg"] = "Anki"
    filetypes["srt"] = "Captions"

    # Add a folders folder as placeholder
    filetypes["placeholder"] = "Folders"

    # Split up contents of Downloads folder and filter out target folders
    download_files, download_folders = categorize_files_folders(download_directory, filetypes)

    # Create folders needed for target
    create_folders(download_directory, filetypes)

    # Move all files and folders in Download folder
    add_keys = move_files(download_directory, download_files, filetypes)
    move_folders(download_directory, download_folders, filetypes)

    # Report missing keys in organize.py
    if add_keys:
        print("Add these keys: ", add_keys)


if __name__ == "__main__":
    try:
        folder = sys.argv[1]
    except:
        folder = "E:\\Downloads"

    main(folder)
