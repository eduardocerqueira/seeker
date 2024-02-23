#date: 2024-02-23T17:02:48Z
#url: https://api.github.com/gists/106824e57cee383eaa87dbc5e56a5a97
#owner: https://api.github.com/users/patrickyee23

import boto3
import os


def check_files():
    s3 = boto3.client("s3")
    fulgent_bucket = "ttam-data-xfer-clinic-fulgent-us-west-2"
    clinic_bucket = "ttam-data-clinic-us-west-2"
    prefix = "wes/raw_data"

    # get all files from fulgent bucket
    response = s3.list_objects(Bucket=fulgent_bucket, Prefix=prefix)
    key_etag = {
        os.path.basename(f["Key"]): f["ETag"].strip('"')
        for f in response["Contents"]
    }

    for fulgent_key, etag in key_etag.items():
        # search clinic bucket
        micronic_barcode = fulgent_key.split("-", 1)[0]
        clinic_prefix = f"{prefix}/{micronic_barcode}"
        response = s3.list_objects(Bucket=clinic_bucket, Prefix=clinic_prefix)
        clinic_keys = [f["Key"] for f in response.get("Contents", [])]
        for f in clinic_keys:
            clinic_key = os.path.basename(f)
            if etag in clinic_key and fulgent_key[-3:] == clinic_key[-3:]:
                # found
                print(f"File {fulgent_key} found in clinic bucket")
                break
        else:
            print(f"File {fulgent_key} not found in clinic bucket: {clinic_keys}")
            print(f"ETag: {etag}")


if __name__ == "__main__":
    check_files()