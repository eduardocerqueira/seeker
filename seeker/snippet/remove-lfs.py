#date: 2021-09-09T17:15:12Z
#url: https://api.github.com/gists/3a314ab3e7ab680d16b5e7eb256cafbd
#owner: https://api.github.com/users/klinki

#!/usr/bin/env python3

"""
This is a simple program that will insert some regular file into the root
commit(s) of history, e.g. adding a file named LICENSE or COPYING to the
first commit.  It also rewrites commit hashes in commit messages to update
them based on these changes.
"""

"""
Please see the
  ***** API BACKWARD COMPATIBILITY CAVEAT *****
near the top of git-filter-repo.
"""

# Technically, this program could be replaced by a one-liner:
#    git filter-repo --force --commit-callback "if not commit.parents: commit.file_changes.append(FileChange(b'M', $RELATIVE_TO_PROJECT_ROOT_PATHNAME, b'$(git hash-object -w $FILENAME)', b'100644'))"
# but let's do it as a full-fledged program that imports git_filter_repo
# anyway...

import re
import argparse
import os
import subprocess
try:
  import git_filter_repo as fr
except ImportError:
  raise SystemExit("Error: Couldn't find git_filter_repo.py.  Did you forget to make a symlink to git-filter-repo named git_filter_repo.py or did you forget to put the latter in your PYTHONPATH?")

parser = argparse.ArgumentParser(
          description='Add a file to the root commit(s) of history')
parser.add_argument('--relevant', metavar="FUNCTION_BODY",
        help=("Python code for determining whether to apply linter to a "
              "given filename.  Implies --filenames-important.  See CALLBACK "
              "below."))

parser.add_argument('--dir', type=os.fsencode,
        help=("Relative-path to file whose contents should be added to root commit(s)"))
lint_args = parser.parse_args()
if not lint_args.dir:
  raise SystemExit("Error: Need to specify the --file option")

# if not args.file:
#   raise SystemExit("Error: Need to specify the --file option")

# fhash = subprocess.check_output(['git', 'hash-object', '-w', args.file]).strip()
# fmode = b'100755' if os.access(args.file, os.X_OK) else b'100644'
# FIXME: I've assumed the file wasn't a directory or symlink...

if not os.path.isdir(lint_args.dir):
    raise SystemExit("--dir must be dir")


blobs_handled = {}
cat_file_process = None

def download_file(content):
    results = re.findall("oid sha256:([a-z0-9]+)", str(content))

    if len(results) == 0:
      return None

    print(results)

    tmpDir = lint_args.dir
    tempFilename = os.path.join(os.path.normpath(tmpDir), os.fsencode(results[0]))
    print ("Reading " + str(tempFilename))

    # Get the new contents
    with open(tempFilename, "rb") as f:
        blob = fr.Blob(f.read())
    # Insert the new file into the filter's stream, and remove the tempfile
    filter.insert(blob)
    os.remove(tempFilename)
    return blob

# def clean_gitattributes():


def fixup_commits(commit, metadata):
    for change in commit.file_changes:
        if change.blob_id in blobs_handled:
            change.blob_id = blobs_handled[change.blob_id]
        elif change.type == b'D':
            continue
        elif not is_relevant(change.filename):
            continue
        else:
            print()
            print (b"Checking " + change.filename)
            print(b"Blob id: " + change.blob_id)

            # Get the old blob contents
            cat_file_process.stdin.write(change.blob_id + b'\n')
            cat_file_process.stdin.flush()

            line = cat_file_process.stdout.readline()
            splitLine = line.split()

            print (splitLine)

            objhash, objtype, objsize = splitLine

            print ("Size is " + str(objsize))

            contents_plus_newline  = cat_file_process.stdout.read(int(objsize)+1)

            if int(objsize) < 120 or int(objsize) > 140:
                continue

            if contents_plus_newline is None:
                continue

            print(b"Replacing " + change.filename)

            # Record our handling of the blob and use it for this change
            blob = download_file(contents_plus_newline)

            blobs_handled[change.blob_id] = blob.id
            change.blob_id = blob.id

#     print(commit.file_changes)

  # if len(commit.parents) == 0:
  #   commit.file_changes.append(fr.FileChange(b'M', args.file, fhash, fmode))
  # FIXME: What if the history already had a file matching the given name,
  # but which didn't exist until later in history?  Is the intent for the
  # user to keep the other version that existed when it existed, or to
  # overwrite the version for all of history with the specified file?  I
  # don't know, but if it's the latter, we'd need to add an 'else' clause
  # like the following:
  #else:
  #  commit.file_changes = [x for x in commit.file_changes
  #                         if x.filename != args.file]

lint_args.filenames_important = True

if lint_args.filenames_important and not lint_args.relevant:
  lint_args.relevant = 'return True'

if lint_args.relevant:
  body = lint_args.relevant
  exec('def is_relevant(filename):\n  '+'\n  '.join(body.splitlines()),
       globals())
  lint_args.filenames_important = True

args = fr.FilteringOptions.default_options()
args.force = True

if lint_args.filenames_important:
  # tmpdir = tempfile.mkdtemp().encode()
  cat_file_process = subprocess.Popen(['git', 'cat-file', '--batch'],
                                      stdin = subprocess.PIPE,
                                      stdout = subprocess.PIPE)
  filter = fr.RepoFilter(args, commit_callback=fixup_commits)
  filter.run()
  cat_file_process.stdin.close()
  cat_file_process.wait()
else:
  filter = fr.RepoFilter(args, blob_callback=lint_non_binary_blobs)
  filter.run()


# fr_args = fr.FilteringOptions.parse_args(['--preserve-commit-encoding',
#                                           '--force',
#                                           '--replace-refs', 'update-no-add'])
# filter = fr.RepoFilter(fr_args, commit_callback=fixup_commits)
# filter.run()
#
