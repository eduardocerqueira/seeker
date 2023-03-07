#date: 2023-03-07T17:11:12Z
#url: https://api.github.com/gists/671a9f971f49661d097aa1655476878e
#owner: https://api.github.com/users/will-moore



import argparse
import locale
import os
import platform
import sys

import omero.clients
from omero.cli import cli_login
from omero.model import ChecksumAlgorithmI
from omero.model import NamedValue
from omero.model.enums import ChecksumAlgorithmSHA1160
from omero.rtypes import rstring, rbool, rlong
from omero_version import omero_version
from omero.gateway import BlitzGateway


def get_files_for_fileset(fs_path):
    filepaths = []
    for path, subdirs, files in os.walk(fs_path):
        for name in files:
            print(os.path.join(path, name))
            # If we want to ignore chunks...
            if ".z" in name or ".xml" in name:
                filepaths.append(os.path.join(path, name))
    return filepaths


def create_fileset(conn, files):
    """Create a new Fileset from local files."""
    fileset = omero.model.FilesetI()
    for f in files:
        fpath, fname, fsize, sha1 = f
        orig = create_original_file(conn, fpath, fname, fsize, sha1)
        entry = omero.model.FilesetEntryI()
        entry.setClientPath(rstring(f))
        # print(dir(entry))
        entry.setOriginalFile(orig)
        fileset.addFilesetEntry(entry)
    return fileset


def create_original_file(conn, path, name, fileSize, shaHast):
    updateService = conn.getUpdateService()
    # create original file, set name, path, mimetype
    originalFile = omero.model.OriginalFileI()
    originalFile.setName(rstring(name))
    originalFile.setPath(rstring(path))
    # if mimetype:
    #     originalFile.mimetype = rstring(mimetype)
    originalFile.setSize(rlong(fileSize))
    # set sha1
    originalFile.setHash(rstring(shaHast))

    chk = omero.model.ChecksumAlgorithmI()
    chk.setValue(rstring(omero.model.enums.ChecksumAlgorithmSHA1160))
    originalFile.setHasher(chk)

    originalFile = updateService.saveAndReturnObject(originalFile, conn.SERVICE_OPTS)
    return originalFile


def getHash(localPath):
    with open(localPath, 'rb') as fileHandle:
        try:
            import hashlib
            hash_sha1 = hashlib.sha1
        except:
            import sha
            hash_sha1 = sha.new
        fileHandle.seek(0)
        h = hash_sha1()
        h.update(fileHandle.read())
        shaHast = h.hexdigest()
    return shaHast


def main(argv):
    parser = argparse.ArgumentParser()

    # source_fileset = 5286802

    parser.add_argument('source', type=int, help='Fileset to use as source of in-place files')
    parser.add_argument('target', type=int, help=('Replace this Fileset...'))
    root_dir = "/data/OMERO/ManagedRepository/demo_52/Blitz-0-Ice.ThreadPool.Server-6/2023-02/27/13-19-26.557/ngff/Tonsil 2.ome.zarr"
    args = parser.parse_args(argv)

    with cli_login() as cli:
        conn = BlitzGateway(client_obj=cli._client)

        source_fileset = conn.getObject("Fileset", args.source)
        if source_fileset is None:
            print ('source Fileset id not found: %s' % args.source)
            sys.exit(1)
        target_fileset = conn.getObject("Fileset", args.target)
        if target_fileset is None:
            print ('target Fileset id not found: %s' % args.target)
            sys.exit(1)

        # For each file in source fileset, create a new Fileset
        file_paths = []
        prefix = source_fileset.templatePrefix
        print("templatePrefix", prefix)
        # for f in source_fileset.listFiles():
        #     print("OriginalFile", f.id, f.path, f.name)
        #     file_paths.append([f.path, f.name, f.size, f.hash])
        for file_path in get_files_for_fileset(root_dir):
            fpath, fname = os.path.split(file_path)
            fsize = os.path.getsize(file_path)
            fhash = getHash(file_path)
            print(fpath, fname, fsize, fhash)
            file_paths.append([fpath, fname, fsize, fhash])

        new_fileset = create_fileset(conn, file_paths)
        new_fileset.templatePrefix = omero.rtypes.rstring(prefix)

        update = conn.getUpdateService()

        new_fileset = update.saveAndReturnObject(new_fileset)
        print("Created new_fileset", new_fileset.id.val)

        for image in target_fileset.copyImages():
            print("Updating image", image.name, image.id)
            img = image._obj
            img.fileset = omero.model.FilesetI(new_fileset.id.val, False)
            conn.getUpdateService().saveObject(img, conn.SERVICE_OPTS)


            # Print the HQL updates we need to update each Pixels to new Fileset file
            # Find the 'key' file that we want to link to in Pixels is the ".zattr" file in root dir
            fname = ".zattrs"
            fpath = root_dir
            pid = image.getPixelsId()
            print(f"""psql -U postgres -d OMERO-server -c "UPDATE pixels SET  name = '{fname}',  path = '{fpath}' where id = {pid}""")

        # delete old fileset...
        # print("Deleting Fileset", old_fs.id)
        # conn.deleteObjects("Fileset", [old_fs.id])


if __name__ == '__main__':
    main(sys.argv[1:])
