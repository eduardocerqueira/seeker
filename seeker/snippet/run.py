#date: 2022-08-31T17:10:28Z
#url: https://api.github.com/gists/ab1cb8a43464a8856bacd49d06a78218
#owner: https://api.github.com/users/dmitry8912

@classmethod
def run(cls, code: str):
    cls.__stack = list()
    cls.__vars = dict()
    cls.__code = list()
    cls.__ip = 0

    cls.__code = code.splitlines()
    while cls.__ip < len(cls.__code):
        # print(cls.__ip)
        cls.run_line(cls.__code[cls.__ip])
        cls.__ip = cls.__ip + 1