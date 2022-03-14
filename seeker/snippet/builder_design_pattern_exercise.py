#date: 2022-03-14T17:04:32Z
#url: https://api.github.com/gists/8652e759f65b171f0f1216688f6e5f1d
#owner: https://api.github.com/users/kziovas

class CodeBuilder:
    def __init__(self, root_name):
        self.root_name = root_name
        self.members = {}

    def add_field(self, type, name):
        self.members[type] = name
        return self

    def __str__(self):
        lines = [f'class {self.root_name}:']
        if not self.members:
            lines.append('  pass')
        else:
            lines.append('  def __init__(self):')
            for k,v in self.members.items():
                lines.append(f'     self.{k} = {v}')
        return '\n'.join(lines)
            
cb = CodeBuilder("Person").add_field("name","").add_field("age","0")
print(cb)