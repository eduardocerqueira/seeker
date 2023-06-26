#date: 2023-06-26T17:06:35Z
#url: https://api.github.com/gists/ca912f6b29a6a20fdd2ea2fe281895db
#owner: https://api.github.com/users/lgray

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import accumulate
from distributed import Client
import dask
import dask_awkward as dak
import dask.array
from dask.diagnostics import ProgressBar


import awkward
import uproot
import numpy
import time

from functools import partial
import gzip
import json
import math
import pprint
import random


def get_steps(
    normed_files,
    maybe_step_size=None,
    align_clusters=False,
    recalculate_seen_steps=False,
):

    nf_backend = awkward.backend(normed_files)
    lz_or_nf = awkward.typetracer.length_zero_if_typetracer(normed_files)

    array = [] if nf_backend != "typetracer" else lz_or_nf
    for arg in lz_or_nf:
        try:
            the_file = uproot.open({arg.file: None})
        except FileNotFoundError as fnfe:
            array.append(None)
            continue

        tree = the_file[arg.object_path]
        num_entries = tree.num_entries

        target_step_size = num_entries if maybe_step_size is None else maybe_step_size

        file_uuid = str(the_file.file.uuid)

        out_uuid = arg.uuid
        out_steps = arg.steps

        if out_uuid != file_uuid or recalculate_seen_steps:
            if align_clusters:
                clusters = tree.common_entry_offsets()
                out = [0]
                for c in clusters:
                    if c >= out[-1] + target_step_size:
                        out.append(c)
                if clusters[-1] != out[-1]:
                    out.append(clusters[-1])
                out = numpy.array(out, dtype="int64")
                out = numpy.stack((out[:-1], out[1:]), axis=1)
            else:
                n_steps = num_entries // target_step_size
                out = numpy.array(
                    [
                        [
                            i * target_step_size,
                            min((i + 1) * target_step_size, num_entries),
                        ]
                        for i in range(n_steps)
                    ],
                    dtype="int64",
                )

            out_uuid = file_uuid
            out_steps = out.tolist()

        array.append(
            {
                "file": arg.file,
                "object_path": arg.object_path,
                "steps": out_steps,
                "uuid": out_uuid,
            }
        )

    if len(array) == 0:
        array = awkward.Array(
            [
                {"file": "junk", "object_path": "junk", "steps": [[]], "uuid": "junk"},
                None,
            ]
        )
        array = awkward.Array(array.layout.form.length_zero_array(highlevel=False))
    else:
        array = awkward.Array(array)

    if nf_backend == "typetracer":
        array = awkward.Array(
            array.layout.to_typetracer(forget_length=True),
        )

    return array


def preprocess(
    fileset,
    maybe_step_size=None,
    align_clusters=False,
    recalculate_seen_steps=False,
    files_per_batch=1,
):
    out_updated = fileset.copy()
    out_available = fileset.copy()
    all_ak_norm_files = {}
    files_to_preprocess = {}
    for name, info in fileset.items():
        norm_files = uproot._util.regularize_files(info["files"], steps_allowed=True)
        for ifile in range(len(norm_files)):
            the_file_info = norm_files[ifile]
            maybe_finfo = info["files"].get(the_file_info[0], None)
            maybe_uuid = (
                None
                if not isinstance(maybe_finfo, dict)
                else maybe_finfo.get("uuid", None)
            )
            norm_files[ifile] += (3 - len(norm_files[ifile])) * (None,) + (maybe_uuid,)
        fields = ["file", "object_path", "steps", "uuid"]
        ak_norm_files = awkward.from_iter(norm_files)
        ak_norm_files = awkward.Array(
            {field: ak_norm_files[str(ifield)] for ifield, field in enumerate(fields)}
        )
        all_ak_norm_files[name] = ak_norm_files

        dak_norm_files = dak.from_awkward(
            ak_norm_files, math.ceil(len(ak_norm_files) / files_per_batch)
        )

        files_to_preprocess[name] = dak.map_partitions(
            get_steps,
            dak_norm_files,
            maybe_step_size=maybe_step_size,
            align_clusters=align_clusters,
            recalculate_seen_steps=recalculate_seen_steps,
        )

    all_processed_files = dask.compute(files_to_preprocess)[0]

    for name, processed_files in all_processed_files.items():
        files_available = {
            item["file"]: {
                "object_path": item["object_path"],
                "steps": item["steps"],
                "uuid": item["uuid"],
            }
            for item in awkward.drop_none(processed_files).to_list()
        }

        files_out = {}
        for proc_item, orig_item in zip(
            processed_files.to_list(), all_ak_norm_files[name].to_list()
        ):
            item = orig_item if proc_item is None else proc_item
            files_out[item["file"]] = {
                "object_path": item["object_path"],
                "steps": item["steps"],
                "uuid": item["uuid"],
            }

        out_updated[name]["files"] = files_out
        out_available[name]["files"] = files_available

    return out_available, out_updated


def apply_to_fileset(fn, preprocessed_fileset):
    out = None

    for name, info in preprocessed_fileset.items():
        metadata = info.get("metadata", {}).copy()
        metadata["dataset"] = name
        events = NanoEventsFactory.from_root(
            info["files"],
            metadata=metadata,
            schemaclass=NanoAODSchema,
            permit_dask=True,
        ).events()

        if out is None:
            out = fn(events)
        else:
            out = accumulate((out, fn(events)))

    return out


def my_analysis(events):
    return {events.metadata["dataset"]: events.Muon.pt}


if __name__ == "__main__":
    # client = Client()
    dataset = {
        "ZJets": {
            "files": {
                "../replicated_sample/nano_dy*.root": "Events",
                "../replicated_sample/nano_dy_101.root": {
                    "object_path": "Events",
                    "steps": [0, 40],
                },
                "../replicated_sample/nano_dy_102.root": {
                    "object_path": "Events",
                    "steps": [0, 40],
                    "uuid": "deadbeef",
                },
            }
        },
        "Data": {"files": {"../replicated_sample/nano_dimuon*.root": "Events"}},
    }
    
    dak_debug = {
        "awkward.optimization.enabled": False,
        "awkward.raise-failed-meta": True,
        "awkward.optimization.on-fail": "raise",
    }

    with dask.config.set({"scheduler": "processes"}), ProgressBar():
        dataset_runnable, dataset = preprocess(
            dataset, maybe_step_size=10, align_clusters=False, files_per_batch=10
        )

        process = partial(
            apply_to_fileset,
            my_analysis,
        )

        result = process(dataset_runnable)

        print(dask.compute(result)[0])

    with gzip.open("normalized_data.json.gz", "wb") as f:
        f.write(bytes(json.dumps(dataset), encoding="ascii"))
