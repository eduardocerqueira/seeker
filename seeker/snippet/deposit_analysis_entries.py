#date: 2024-02-09T17:06:06Z
#url: https://api.github.com/gists/eda0181cfa7869936e312cb6919106c4
#owner: https://api.github.com/users/aheinzel

#!/usr/bin/env python3

import sys
import requests
import re
import os.path

#user and pass for auth
EGA_USER=""
EGA_PASS=""

#basic parameters regarding submission/experiment
SUBMISSION_ID="7347"
STUDY_ID=380
EXPERIMENT_ID=7016
SUBMISSION_ID=7347
FILE_SUFFIX="_TRB_mig_cdr3_clones_result.csv.c4gh"
ANALYSIS_TITLE="TCRB clonotype table"
ANALYSIS_DESCRIPTION="TCRB clonotype table (Takara Immune Profiler)"

#base uris
IDP_URL='https: "**********"
SP_URL='https://submission.ega-archive.org/api'



def _new_analysis_entry(sample_id, file_id):
    return {
        "title":ANALYSIS_TITLE,
        "description":ANALYSIS_DESCRIPTION,
        "analysis_type":"SAMPLE PHENOTYPE",
        "files":[file_id],
        "study_provisional_id": STUDY_ID,
        "experiment_provisional_ids": [EXPERIMENT_ID],
        "sample_provisional_ids": [sample_id],
        "submission_provisional_id": SUBMISSION_ID
   }


def _nh(bearer):
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer}"
    }


def auth():
    payload = {
        "grant_type": "**********"
        "client_id": "sp-api",
        "username": EGA_USER,
        "password": "**********"
    }
  
    r = requests.post(IDP_URL, data = payload)
    return r.json()["access_token"]


def _make_sample_id_lookup(samples):
    lookup = dict()
    for s in samples:
        assert(s["alias"] not in lookup)
        lookup[s["alias"]] = s["provisional_id"]
    
    return lookup


def retrieve_files(bearer):
    r = requests.get(f"{SP_URL}/files?status=inbox", headers = _nh(bearer))
    return r.json()


def retrieve_samples(submission_id, bearer):
    r = requests.get(f"{SP_URL}/submissions/{submission_id}/samples", headers = _nh(bearer))
    return r.json()


def associate_files_with_samples(files, sample_id_lookup, get_sample_alias = lambda _: _.replace(FILE_SUFFIX, "")):
    for file in files:
        file_name = os.path.basename(file["relative_path"])
        sample_alias = get_sample_alias(file_name)
        if sample_alias not in sample_id_lookup:
            print(f"skipping file {file_name} no corresponding sample", file = sys.stderr)
        else:
            sample_id = sample_id_lookup[sample_alias]
            yield _new_analysis_entry(sample_id, file["provisional_id"])


def deposit_analysis_entries(analysis_entries, bearer):
    r = requests.post(f"{SP_URL}/submissions/{SUBMISSION_ID}/analyses", headers = _nh(bearer), json = analysis_entries)
    return r.status_code == 200 #yes at the time of writing it was returning 200


def main():
    bearer = auth()
    files = list(
        filter(
            lambda _: _["relative_path"].endswith(FILE_SUFFIX),
            retrieve_files(bearer)
        )
    )
    samples = retrieve_samples(SUBMISSION_ID, bearer)
    sample_id_lookup = _make_sample_id_lookup(samples)
    analysis_entries = list(associate_files_with_samples(files, sample_id_lookup))
    assert(deposit_analysis_entries(analysis_entries, bearer))
    



main()ies(analysis_entries, bearer))
    



main()