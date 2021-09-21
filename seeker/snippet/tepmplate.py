#date: 2021-09-21T16:53:13Z
#url: https://api.github.com/gists/be9a34e97d8903ccd52701f367b61802
#owner: https://api.github.com/users/KoliosterNikolayIliev

import re


class TemplateEngineError(Exception):
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass


class TemplateEngine:
    """
    Putting __matches and __variables as protected attributes in the constructor is:
    a)bad (coupling)
    b)irrelevant 
    c)not so bad
    """
    def __init__(self, template):
        self.template = template
        self.__matches = re.findall('{\s*[a-zA-z]*\s*}', self.template)
        self.__variables = [x.strip('{} ') for x in self.__matches]

    def render(self, **context):
        result = self.template.replace('{{', '{').replace('}}', '}')

        for key, variable in enumerate(self.__variables):
            if variable not in context.keys():
                raise TemplateEngineError('Not all variables, present in `template`, have values in `context`')
            result = result.replace(self.__matches[key], context[variable])

        return result

    def extract_variables(self):
        return self.__variables
