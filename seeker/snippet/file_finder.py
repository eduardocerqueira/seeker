#date: 2022-10-04T17:07:28Z
#url: https://api.github.com/gists/72ef0ec3f38ce3d4e83d51ee8fa06e72
#owner: https://api.github.com/users/DanielCreeklear

import glob
import os

class FileFinder:
    @staticmethod
    def get_all_files_in(path: str, filter=None, number_children=0) -> list:
        wildcard_directory = r'\*'
        additional_path = wildcard_directory * number_children
        if filter is not None:
            filter_file = rf'\*.{filter}'
        else:
            filter_file = ''

        try:
            return [path for path in glob.glob(path + additional_path + filter_file) if os.path.isfile(path)]
        except Exception as ex:
            print(f'Não foi possível obter os arquivos em: {path + additional_path + filter_file} [{ex}]')
            return []