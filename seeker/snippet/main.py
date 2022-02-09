#date: 2022-02-09T17:12:36Z
#url: https://api.github.com/gists/4c241aae7d38307e5d8f02af1fcb60b8
#owner: https://api.github.com/users/mypy-play

class ExtraValuemixin:
    def __init__(self, value, *args, **kwargs):
        super().__init__(value, *args, **kwargs)

    def retrieve_extra_value(self):
        return self.value


class ParentObj:
    def __init__(self, value):
        self.value = value


class ChildObj(ExtraValuemixin, ParentObj):
    pass


obj = ChildObj(value=5)

print(obj.retrieve_extra_value())