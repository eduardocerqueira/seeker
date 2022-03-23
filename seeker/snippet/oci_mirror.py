#date: 2022-03-23T16:57:14Z
#url: https://api.github.com/gists/3222408bd46ab3114e5b384f20252c10
#owner: https://api.github.com/users/wolfv

from conda_package_handling import api as cph_api
from tempfile import TemporaryDirectory
import pathlib
import os
import subprocess
import shutil
import json
import requests
import tarfile

info_archive_media_type = "application/vnd.conda.info.v1.tar+gzip"
info_index_media_type = "application/vnd.conda.info.index.v1+json"
package_tarbz2_media_type = "application/vnd.conda.package.v1"
package_conda_media_type = "application/vnd.conda.package.v2"

CACHE_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "cache"

class Layer:
    def __init__(self, file, media_type):
        self.file = file
        self.media_type = media_type

class ORAS:
    def __init__(self, base_dir="."):
        self.exec = 'oras'
        self.base_dir = pathlib.Path(base_dir)

    def run(self, args):
        return subprocess.run([self.exec] + args, cwd=self.base_dir)

    def pull(self, location, subdir, package_name, media_type):
        name, version, build = package_name.rsplit('-', 2)
        location = f'{location}/{subdir}/{name}:{version}-{build}'
        args = ['pull', location, '--media-type', media_type]

        self.run(args)

    def push(self, target, tag, layers, config=None):
        layer_opts = [f'{str(l.file)}:{l.media_type}' for l in layers]
        dest = f'{target}:{tag}'
        args = ["push", dest] + layer_opts

        return self.run(args)

class SubdirAccessor:

    def __init__(self, location, subdir, base_dir='.'):
        self.loc = location
        self.subdir = subdir
        self.oras = ORAS(base_dir=base_dir)

    def get_index_json(self, package_name):
        self.oras.pull(self.loc, self.subdir, package_name, info_index_media_type)
        with open(pathlib.Path(package_name) / 'info' / 'index.json') as fi:
            return json.load(fi)

    def get_info(self, package_name):
        self.oras.pull(self.loc, self.subdir, package_name, info_archive_media_type)
        return tarfile.open(pathlib.Path(package_name) / 'info.tar.gz', 'r:gz')

    def get_package(self, package_name):
        self.oras.pull(self.loc, self.subdir, package_name, package_tarbz2_media_type)
        return package_name + '.tar.bz2'


def compress_folder(source_dir, output_filename):
    return subprocess.check_output(f'tar -cvzf {output_filename} *', cwd=source_dir, shell=True)

# def extract(fn, dest_dir=None, components=None, prefix=None):
def get_package_name(path_to_archive):
    fn = pathlib.Path(path_to_archive).name
    if fn.endswith('.tar.bz2'):
        return fn[:-8]
    elif fn.endswith('.conda'):
        return fn[:-6]
    else:
        raise RuntimeError("Cannot decipher package type")

def prepare_metadata(path_to_archive, upload_files_directory):
    package_name = get_package_name(path_to_archive)

    dest_dir = pathlib.Path(upload_files_directory) / package_name
    print(dest_dir)
    dest_dir.mkdir(parents=True)

    with TemporaryDirectory() as temp_dir:
        cph_api.extract(str(path_to_archive), temp_dir, components=['info'])
        index_json = os.path.join(temp_dir, "info", "index.json")
        info_archive = os.path.join(temp_dir, 'info.tar.gz')
        compress_folder(os.path.join(temp_dir, 'info'), os.path.join(temp_dir, 'info.tar.gz'))

        (dest_dir / 'info').mkdir(parents=True)
        shutil.copy(info_archive, dest_dir / 'info.tar.gz')
        shutil.copy(index_json, dest_dir / 'info' / 'index.json')

    for x in pathlib.Path(dest_dir).iterdir():
        print(x)

def upload_conda_package(path_to_archive, host):
    path_to_archive = pathlib.Path(path_to_archive)
    package_name = get_package_name(path_to_archive)

    with TemporaryDirectory() as upload_files_directory:
        shutil.copy(path_to_archive, upload_files_directory)

        prepare_metadata(path_to_archive, upload_files_directory)

        if path_to_archive.name.endswith('tar.bz2'):
            layers = [Layer(path_to_archive.name, package_tarbz2_media_type)]
        else:
            layers = [Layer(path_to_archive.name, package_conda_media_type)]
        metadata = [Layer(f"{package_name}/info.tar.gz", info_archive_media_type),
                    Layer(f"{package_name}/info/index.json", info_index_media_type)]


        for x in pathlib.Path(upload_files_directory).rglob("*"):
            print(x)

        oras = ORAS(base_dir=upload_files_directory)

        name = package_name.rsplit('-', 2)[0]
        version_and_build = '-'.join(package_name.rsplit('-', 2)[1:])

        with open(pathlib.Path(upload_files_directory) / package_name / 'info' / 'index.json', 'r') as fi:
            j = json.load(fi)
            subdir = j["subdir"]


        oras.push(f'{host}/{subdir}/{name}', version_and_build, layers + metadata)

def get_repodata(channel, subdir):
    repodata = CACHE_DIR / channel / subdir / "repodata.json"
    if repodata.exists():
        return repodata
    repodata.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"https://conda.anaconda.org/{channel}/{subdir}/repodata.json", allow_redirects=True)
    with open(repodata, 'w') as fo:
        fo.write(r.text)

    return repodata

def get_github_packages(location, channel, subdir=''):
    org = location.split('/', 1)
    # api_url = f'https://api.github.com/orgs/{org}/packages'
    headers = {'accept': 'application/vnd.github.v3+json'}
    api_url = f'https://api.github.com/users/wolfv/packages'
    r = requests.get(api_url, headers=headers)
    print(r.json())

if __name__ == '__main__':

    channel = 'conda-forge'
    subdir = 'osx-arm64'
    repodata_fn = get_repodata(channel, subdir)

    # get_github_packages('ghcr.io/wolfv', '')
    # exit(0)

    # with open(repodata_fn) as fi:
    #     j = json.load(fi)

    # for key, package in j["packages"].items():
    #     if package["name"] == 'xtensor':
    #         print("Loading ", key)

    #         r = requests.get(f"https://conda.anaconda.org/{channel}/{subdir}/{key}", allow_redirects=True)
    #         with open(key, 'wb') as fo:
    #             fo.write(r.content)
    #         upload_conda_package(key, 'ghcr.io/wolfv')

    subdir = SubdirAccessor('ghcr.io/wolfv', 'osx-arm64')
    index = subdir.get_index_json('xtensor-0.21.10-h260d524_0')
    print(index)

    with subdir.get_info('xtensor-0.21.10-h260d524_0') as fi:
        paths = json.load(fi.extractfile('paths.json'))
        print(paths)
