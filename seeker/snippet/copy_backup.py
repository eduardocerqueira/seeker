#date: 2024-02-23T17:02:48Z
#url: https://api.github.com/gists/106824e57cee383eaa87dbc5e56a5a97
#owner: https://api.github.com/users/patrickyee23

import boto3
import os


def copy_files():
    s3 = boto3.client("s3")
    fulgent_bucket = "ttam-data-xfer-clinic-fulgent-us-west-2"
    clinic_bucket = "ttam-data-clinic-us-west-2"
    prefix = "wes/raw_data"
    backup_prefix = f"backup/{prefix}"

    # get all micronic_barcode from fulgent bucket
    response = s3.list_objects(Bucket=fulgent_bucket, Prefix=prefix)
    micronic_barcodes = set([os.path.basename(f["Key"]).split("-", 1)[0] for f in response.get("Contents", [])])

    for micronic_barcode in micronic_barcodes:
        # first try to find back up from the backup prefix
        response = s3.list_objects(Bucket=fulgent_bucket, Prefix=f"{backup_prefix}/{micronic_barcode}")
        keys = [f["Key"] for f in response.get("Contents", [])]
        pdf_found = any([key.endswith(".pdf") for key in keys])
        tsv_found = any([key.endswith(".tsv") for key in keys])
        if pdf_found and tsv_found:
            for key in keys:
                if key.endswith(".pdf") or key.endswith(".tsv"):
                    new_key = key.replace(backup_prefix, prefix)
                    print(f"aws s3 cp s3://{fulgent_bucket}/{key} s3://{fulgent_bucket}/{new_key}")
            continue

        # if not found, try to find from the clinic bucket
        clinic_prefix = f"{prefix}/{micronic_barcode}"
        response = s3.list_objects(Bucket=clinic_bucket, Prefix=clinic_prefix)
        keys = [f["Key"] for f in response.get("Contents", [])]
        pdf_found = any([key.endswith(".pdf") for key in keys])
        tsv_found = any([key.endswith(".tsv") for key in keys])
        if pdf_found and tsv_found:
            for key in keys:
                if key.endswith(".pdf") or key.endswith(".tsv"):
                    new_key = key.replace(f"/{micronic_barcode}/", "/")
                    name, ext = os.path.splitext(new_key)
                    name = "-".join(name.split("-")[:-1])
                    new_key = f"{name}{ext}"
                    print(f"aws s3 cp s3://{clinic_bucket}/{key} s3://{fulgent_bucket}/{new_key}")
            continue

        print(f"Files not found for {micronic_barcode}")


if __name__ == "__main__":
    copy_files()