#date: 2023-03-16T16:56:48Z
#url: https://api.github.com/gists/45b7a176a537b4d80cb1f14d6cc28dba
#owner: https://api.github.com/users/mayasaxena

"""Extract .md files matching certain criteria into a folder for publishing

Required packages:
- python-frontmatter
- send2trash
- case-converter
"""
import os
import sys
from pathlib import Path
import frontmatter
from send2trash import send2trash
from caseconverter import kebabcase

# All paths assume our current working directory is at the root of the obsidian project
PUBLISH_FLAG = "publish"
PATH_FLAG = "exclude_path"
EXCLUDE = ["/.", "Templates"] # Ignore folders starting with dots and templates
PUBLISH_ALLOWED = ["Snippets"]
SOURCE_FILES = Path("../")
FROZEN_FOLDER = Path("../Frozen")
DESTINATION_FOLDER = Path("docs/")

DRY_RUN=False

def run():
    """Run the script"""
    modified = []
    published = []
    for source_path, publish, exclude_path in get_published_notes():
        if type(publish) == bool: # Regular note, create symlink mirroring path
            destination_path = get_destination_path(source_path, exclude_path)
        elif source_path.is_relative_to(FROZEN_FOLDER): # Frozen note, create symlink from publish
            destination_path = get_destination_path(publish, frozen=True)

        if destination_path is None:
            print("Invalid destination path", destination_path)
            continue
        else:
            if publish is not False:
                published.append(destination_path)

        # Create the destination folder recursively if it is missing
        if publish:
            if m := create_folder_if_necessary_for(destination_path): modified.append(m)

        if not destination_path.is_file(): 
            if publish is not False: # Create symlink if it does not exist and publish is not false
                modified.append(create_symlink(source_path, destination_path))
        else:
            if publish is False: # Delete symlink if it exists and publish is False
                modified.append(trash(destination_path))
            elif publish is True and destination_path not in destination_path.parent.iterdir(): # Rename and fix case
                # THIS DOESN'T CATCH FOLDER NAMES
                modified.append("* " + str(destination_path))

    if DRY_RUN:
        print("publishing: ")
        [print(str(p)) for p in published]
    
    return modified + delete_stale(DESTINATION_FOLDER, published)

def get_published_notes():
    """Generator yielding paths and content of notes that are marked as published"""
    for item in SOURCE_FILES.rglob("*.md"):
        if should_include(item):
            with open(item) as f:
                metadata, content = frontmatter.parse(f.read())
                publish = metadata.get(PUBLISH_FLAG)
                if should_publish(item, content, publish):
                    yield item, publish, metadata.get(PATH_FLAG)

def should_publish(item, content, publish):
    return (publish is not None
            and ("```dataview" not in content or any(allowed in str(item) for allowed in PUBLISH_ALLOWED)))

def should_include(item: Path):
    return (not any(exclude in str(item.parent) for exclude in EXCLUDE)
            and item.is_file()
            and not item.name.startswith("."))

def get_destination_path(source_path, exclude_path, frozen=False):
    if frozen:
        publish_path = source_path
    else:
        try:
            publish_path = source_path.relative_to("../" + (exclude_path or ""))
        except ValueError as ve:
            print(ve)
            publish_path = source_path.relative_to("../" + (""))
        except:
            return
        
    return kebab(DESTINATION_FOLDER / publish_path)

def kebab(path: Path):
    return Path(str(path.parent / kebabcase(path.stem)).lower()).with_suffix(path.suffix)

def create_folder_if_necessary_for(destination_path):
    destination_folder = "/".join(str(destination_path).split("/")[:-1])
    if not Path(destination_folder).is_dir():
        if not DRY_RUN:
            os.makedirs(destination_folder, exist_ok=True)
        return "+ " + str(destination_folder)

def create_symlink(source_path, destination_path):
    relative_source_path = os.path.relpath(source_path, destination_path.parent)
    if not DRY_RUN:
        destination_path.symlink_to(relative_source_path)
    return "+ " + str(destination_path)

def trash(destination_path):
    if not DRY_RUN:
        send2trash(destination_path)
    return "- " + str(destination_path)

def delete_stale(root, published):
    modified = []
    for p in Path(root).glob('**/*'):
        if p.is_dir() and should_check(p):
            modified += delete_stale_files(p, published)
        elif p.is_file and should_delete_file(p, published):
            if not DRY_RUN or len([m for m in modified if str(p) in m]) == 0:
                modified.append(trash(p))
    return modified

def should_check(p):
    return p.is_dir() and "assets" not in str(p)

def delete_stale_files(folder, published):
    if not folder.is_dir(): return []
    modified = []
    files = list(folder.iterdir())
    remove = [fname for fname in files if should_delete_file(fname, published)]
    if len(files) == 0 or (len(files) == 1 and is_dsstore(files[0].name)) or files == remove:
        if not DRY_RUN or len([p for p in published if folder in p.parents]) == 0:
            modified.append(trash(folder))
    else:
        for fname in remove:
            modified.append(trash(fname))
    return modified

def should_delete_file(fname, published):
    return ((fname.suffix == ".md" and "index" not in str(fname))
            and (fname.is_symlink() or fname.is_file()) 
            and not is_dsstore(fname) 
            and str(fname).lower() not in (str(p).lower() for p in published))

def is_dsstore(fname):
    return ".DS_Store" in str(fname)

def log(modified, symbol="", name="modified"):
    matches = len([x for x in modified if (symbol + " ") in x])
    inset = "" if symbol == "" else "  "
    if matches > 0:
        print(inset + str(matches), "file" + ("s" if matches > 1 else ""), name)

if __name__ == "__main__":
    DRY_RUN = input("Dry run? (y/n) ").lower().startswith("y")
    modified = run()
    if len(modified) > 0:
        added = [x for x in modified if "+ " in x]
        removed = [x for x in modified if "- " in x]
        renamed = [x for x in modified if "* " in x]
        print("Files modified, make sure to git add")
        [print(" ", m) for m in modified]
        log(modified)
        log(modified, "+", "added")
        log(modified, "-", "removed")
        log(modified, "*", "should be manually renamed")
        sys.exit(1)