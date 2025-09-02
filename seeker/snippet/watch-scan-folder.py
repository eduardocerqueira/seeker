#date: 2025-09-02T17:02:39Z
#url: https://api.github.com/gists/1f6e6acf1e0d1b944433cb042ad308b0
#owner: https://api.github.com/users/lnsy-dev

import os
import shutil
import time
import sys
import traceback
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- Dependencies Check ---
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("Error: 'docling' library not found.")
    print("Please install it using: pip install docling-ai")
    sys.exit(1)

# --- Configuration ---
HOME = os.path.expanduser("~")
TO_SCAN_DIR = os.path.join(HOME, 'shared', 'to-scan')
SCANNED_DIR = os.path.join(HOME, 'shared', 'scanned')

# --- State Management ---
# A set to track files currently being processed to avoid race conditions from multiple events.
PROCESSING = set()

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)
        self.saved_path = None

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)

def is_image(filename):
    """Checks if a file is an image based on its extension."""
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))

def process_image(image_path):
    """
    Processes a single image file by changing to its directory, running conversion,
    and then moving the files to the scanned directory.
    """
    if image_path in PROCESSING:
        print(f"-> Skipping {image_path}, already in processing queue.")
        return

    if not os.path.exists(image_path):
        print(f"-> Skipping {image_path}, it no longer exists.")
        return

    print(f"--- Starting processing for {image_path} ---")
    PROCESSING.add(image_path)

    try:
        image_dir = os.path.dirname(image_path)
        image_filename = os.path.basename(image_path)

        # Temporarily change the current working directory to the image's directory
        with cd(image_dir):
            print(f"-> CWD temporarily changed to: {os.getcwd()}")
            
            print("-> Initializing DocumentConverter...")
            converter = DocumentConverter()
            print("-> DocumentConverter initialized.")

            print(f"-> Starting conversion for '{image_filename}'...")
            result = converter.convert(image_filename) # Use filename only, as in the example script
            print("-> Conversion successful.")

            print("-> Exporting to Markdown...")
            text = result.document.export_to_markdown()
            print("-> Markdown export successful.")

            markdown_filename = os.path.splitext(image_filename)[0] + '.md'

            print(f"-> Writing Markdown content to '{markdown_filename}' in CWD...")
            with open(markdown_filename, 'w', encoding='utf-8') as md_file:
                md_file.write(text + '\n\n')
                md_file.write(f"![[{image_filename}]]")
            print("-> Markdown file written successfully.")

        # CWD is now restored. Proceed with moving files using absolute paths.
        print(f"-> CWD restored to: {os.getcwd()}")

        abs_markdown_path = os.path.join(image_dir, markdown_filename)
        dest_image_path = os.path.join(SCANNED_DIR, image_filename)
        dest_markdown_path = os.path.join(SCANNED_DIR, markdown_filename)

        print(f"-> Moving {image_path} to {dest_image_path}...")
        shutil.move(image_path, dest_image_path)
        print("-> Image file moved.")

        print(f"-> Moving {abs_markdown_path} to {dest_markdown_path}...")
        shutil.move(abs_markdown_path, dest_markdown_path)
        print("-> Markdown file moved.")

    except Exception as e:
        print(f"-> ERROR: An exception occurred while processing {image_path}: {e}")
        traceback.print_exc()
    finally:
        PROCESSING.remove(image_path)
        print(f"--- Finished processing for {image_path} ---")

class ScanFolderHandler(FileSystemEventHandler):
    """
    Handles file system events for the to-scan directory.
    """
    def on_any_event(self, event):
        """Log all events."""
        print(f"Watchdog event: {event.event_type} on path: {event.src_path}")

    def on_modified(self, event):
        """
        Called when a file or directory is modified.
        This is often a more reliable event than 'on_created' for file drops.
        """
        print(f"-> Handling 'modified' event for: {event.src_path}")
        if event.is_directory:
            print("-> Path is a directory, ignoring.")
            return

        # Process only if the file is an image and exists in the root of the to-scan folder
        if is_image(event.src_path) and os.path.dirname(event.src_path) == TO_SCAN_DIR:
            print(f"-> File is an image in the target directory. Queuing for processing.")
            time.sleep(1) # A brief pause to ensure the file is fully written
            process_image(event.src_path)
        else:
            print("-> File is not an image or not in the root scan directory, ignoring.")

def main():
    """
    Main function to set up and start the directory watcher.
    """
    for path in [TO_SCAN_DIR, SCANNED_DIR]:
        if not os.path.isdir(path):
            print(f"Error: Directory '{path}' not found.")
            sys.exit(1)

    print(f"Starting file watcher for: {TO_SCAN_DIR}")
    print(f"Processed files will be moved to: {SCANNED_DIR}")
    print("Press Ctrl+C to stop the script.")

    event_handler = ScanFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, TO_SCAN_DIR, recursive=True)
    observer.start()
    print(">>> Watchdog observer started. Waiting for file events... <<<")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping file watcher...")
        observer.stop()
    
    observer.join()
    print("File watcher stopped.")

if __name__ == "__main__":
    main()
