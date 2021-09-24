#date: 2021-09-24T17:07:08Z
#url: https://api.github.com/gists/89975b5aaaac65217c6204b886fae975
#owner: https://api.github.com/users/Terf

import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


info = {} # key for format string => dicom series
info_keys = {} # format string => key for string


def t1_key(acq):
    s = 'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-%s_run-{item:02d}_T1w' % acq.lower()
    if s not in info_keys:
        k = create_key(s)
        info_keys[s] = k
        info[info_keys[s]] = []
    return info_keys[s]
def flair_key(acq):
    s = 'sub-{subject}/{session}/anat/sub-{subject}_{session}_acq-%s_run-{item:02d}_FLAIR' % acq.lower()
    if s not in info_keys:
        k = create_key(s)
        info_keys[s] = k
        info[info_keys[s]] = []
    return info_keys[s]
def dwi_key(acq):
    s = 'sub-{subject}/{session}/dwi/sub-{subject}_{session}_acq-%s_run-{item:02d}_dwi' % acq.lower()
    if s not in info_keys:
        k = create_key(s)
        info_keys[s] = k
        info[info_keys[s]] = []
    return info_keys[s]

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    for s in seqinfo:
        """
        The namedtuple `s` contains the following fields:

        * total_files_till_now
        * example_dcm_file
        * series_id
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        """

        if s.is_derived:
            continue

        protocol_name = s.protocol_name.upper().replace("_", "").replace("-", "").replace(" ", "")
        if 'SPINE' in protocol_name or 'HEAD' in protocol_name or 'POST' in protocol_name:
            continue

        if 'SAG' in protocol_name and 'T1' in protocol_name and 'MPRAGE' in protocol_name:
            key = t1_key(protocol_name)
            info[key].append(s.series_id)
        elif 'DTI' in protocol_name:
            key = dwi_key(protocol_name)
            info[key].append(s.series_id)
        elif 'T2' in protocol_name and 'FLAIR' in protocol_name:
            key = flair_key(protocol_name)
            info[key].append(s.series_id)

    return info
