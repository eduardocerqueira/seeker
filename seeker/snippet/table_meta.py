#date: 2024-02-06T16:53:56Z
#url: https://api.github.com/gists/06e5375198e04bfaccc64421031adf81
#owner: https://api.github.com/users/filipamiralopes

import yaml

class ColumnMeta:
    def __init__(self, name, type, description=None):
        self.name = name
        self.type = type
        self.description = description

    def from_dict(value: dict):
        return ColumnMeta(name=value['name'], type=value['type'], description=value.get('description'))


class TableMeta:
    def __init__(self, name, columns, description=None):
        self.name = name
        self.columns = columns 
        self.description = description

    def from_dict(value):
        columns = [ColumnMeta.from_dict(v) for v in value['columns']]
        return TableMeta(name=value['name'], columns=columns, description=value.get('description'))

    def from_file(path):
        file = open(path, 'r')
        yaml_to_dict = yaml.safe_load(file.read())
        try:
            file.close()
        except Exception:
            pass
        return TableMeta.from_dict(yaml_to_dict)