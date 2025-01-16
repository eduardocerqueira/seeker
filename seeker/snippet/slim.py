#date: 2025-01-16T16:46:54Z
#url: https://api.github.com/gists/ee4d509c0b1d94040fc4feef0c034512
#owner: https://api.github.com/users/pombredanne

#!/usr/bin/env python

# -*- coding: utf-8 -*-

#

# Copyright (c) nexB Inc. and others. All rights reserved.

# SPDX-License-Identifier: Apache-2.0

# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.

# See https://github.com/aboutcode-org for support or download.

# See https://aboutcode.org for more information about AboutCode OSS projects.

#

import argparse

import hashlib

import tarfile

import shutil

from pathlib import Path

from dataclasses import dataclass

@dataclass

class Layer:

    path: Path

    layer_id: str

    old_sha256: str

    new_sha256: str = ""

    layer_number: int = 0

    @classmethod

    def from_tar(cls, layer_tar):

        return cls(

            path=layer_tar,

            layer_id=layer_tar.parent.name,

            old_sha256=sha2(layer_tar)

        )

    def empty_tarball(self):

        """Empty / Wipe clean the layer tarball"""

        with tarfile.TarFile.open(self.path, "w"):

            pass

        self.new_sha256 = sha2(self.path)

def sha2(path):

    return hashlib.sha256(path.read_bytes()).hexdigest()

def make_slim(image_tarball, layer_ids_to_empty):

    """Make an image slimmer, emptying the tarball of some layers"""

    extracted = Path("extracted")

    print(f"Deleting previous {extracted!r}")

    shutil.rmtree(path=extracted, ignore_errors=True)

    extracted.mkdir(exist_ok=True)

    extracted = extracted.absolute()

    print(f"Extracting image to {extracted!r}")

    shutil.unpack_archive(Path(image_tarball), extract_dir=extracted)

    # process layers

    layers = [Layer.from_tar(layer_tar)for layer_tar in extracted.rglob("**/layer.tar")]

    to_empty = [l for l in layers if l.layer_id in layer_ids_to_empty]

    print(f"Emptying Layers:")

    for layer in to_empty:

        print(f"  {layer!r}")

        layer.empty_tarball()

    # update config

    old_config_file = [x for x in extracted.glob("*.json") if x.name != "manifest.json"][0]

    old_config_sha256 = sha2(old_config_file)

    old_config = old_config_file.read_text()

    for layer in to_empty:

        old_config = old_config.replace(f"sha256:{layer.old_sha256}", f"sha256:{layer.new_sha256}")

    old_config_file.write_text(old_config)

    new_config_sha256 = sha2(old_config_file)

    new_config_file = extracted / f"{new_config_sha256}.json"

    old_config_file.rename(new_config_file)

    # update manifest

    manifest_file = extracted / "manifest.json"

    manifest = manifest_file.read_text()

    manifest = manifest.replace(old_config_sha256, new_config_sha256)

    manifest_file.write_text(manifest)

    # recreate image tarball

    shutil.make_archive(

        base_name=f"{image_tarball.stem}-slim",

        format="tar",

        root_dir=extracted,

        base_dir="",

    )

def slimify():

    description = """

    Make an image slim, replacing some layer tarballs by empty tarballs. The new image will have a "-slim" name suffix.

    """

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(

        "-i",

        "--image",

        dest="image_tarball",

        type=Path,

        required=True,

        metavar="FILE",

        help="Path to an image tarball, exported using 'docker save'",

    )

    parser.add_argument(

        "-l",

        "--layer-ids-to-skip",

        dest="layers_file",

        type=Path,

        required=True,

        metavar="FILE",

        help="Path to a file with one layer id to skip per line. "

        "The layer id is the name of the directory that contains a 'layer.tar' tarball.",

    )

    args = parser.parse_args()

    image_tarball = args.image_tarball

    layers_file = args.layers_file

    print(f"Slimifying {image_tarball!r} to '{image_tarball}.slim' skipping layers in {layers_file!r}")

    layer_ids = layers_file.read_text().strip().split()

    make_slim(image_tarball=image_tarball, layer_ids_to_empty=layer_ids)

if __name__ == "__main__":

    slimify()

