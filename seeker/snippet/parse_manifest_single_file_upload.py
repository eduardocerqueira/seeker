#date: 2024-10-02T16:52:17Z
#url: https://api.github.com/gists/c628b6cce749d300790f0cd6f002ea3c
#owner: https://api.github.com/users/BryanFauble

import os
import asyncio
import synapseclient
import synapseclient.models
import synapseutils

syn = synapseclient.Synapse(debug=True)
syn.login()

# Synapse ID for upload location. Sasha/Jineta will send a reference table so you will know which ID to use for which project files.
synapseID = "syn62860087"

# Change this to be the absolute path to local directory containing data to upload
PROJECT_DIRECTORY_PATH = "/projects/ecgc-analysis/CP01JHU513"

# This script will create the manifest file requrid for uploads in your home directory
PATH_TO_MANIFEST_FILE = os.path.expanduser(os.path.join("~", "manifest-for-upload.tsv"))

# Step 2: Create a manifest TSV file to upload data in bulk
# Note: When this command is run it will re-create your directory structure within
# Synapse. Be aware of this before running this command.
# If folders with the exact names already exists in Synapse, those folders will be used.
synapseutils.generate_sync_manifest(
    syn=syn,
    directory_path=PROJECT_DIRECTORY_PATH,
    parent_id=synapseID,
    manifest_path=PATH_TO_MANIFEST_FILE,
)

# Replaced with the below function
# synapseutils.syncToSynapse(
#     syn=syn, manifestFile=PATH_TO_MANIFEST_FILE, sendMessages=False
# )


async def main() -> None:
    """
    Parse a manifest file and upload the files with a few fields. Note that this
    does not support everything `syncToSynapse` does. For example annotations are not
    set on the files. The purpose of this function is to remove as many 'moving parts'
    of the upload from manifest process.
    """

    manifest_dataframe = synapseutils.sync.readManifestFile(syn, PATH_TO_MANIFEST_FILE)

    total = len(manifest_dataframe)
    success = 0
    failure = 0
    print(f"Uploading {total} files")

    for _, row in manifest_dataframe.iterrows():
        try:
            print(f"Uploading {row['path']} to {row['parent']}")
            file = synapseclient.models.File(
                path=row["path"],
                parent_id=row["parent"],
                name=row["name"] if "name" in row else None,
                id=row["id"] if "id" in row else None,
            )
            await file.store_async(synapse_client=syn)
            print(f"Uploaded {row['path']} to {row['parent']}")
            success += 1
        except Exception as e:
            print(f"Failed to upload {row['path']} to {row['parent']}: {e}")
            failure += 1

    print(f"Uploaded {success} files, failed to upload {failure} files, total {total} files")

asyncio.run(main())
