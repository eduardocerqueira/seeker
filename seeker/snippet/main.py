#date: 2025-11-03T16:59:44Z
#url: https://api.github.com/gists/cea80a8ef50deb990eaf78e96e0ef54e
#owner: https://api.github.com/users/mypy-play

from typing import ClassVar, Protocol, runtime_checkable

@runtime_checkable
class DataProviderDownload(Protocol):
    name: ClassVar[str]
    
    def download(self, x: int) -> str:
        ...

@runtime_checkable
class DataProviderUpload(Protocol):
    name: ClassVar[str]

    def upload(self, x: int) -> str:
        ...

class DataProvider(DataProviderDownload, DataProviderUpload, Protocol):
    ...

class MyGoodDataProvider(DataProvider):
    name = "good"

    def download(self, x: int) -> str:
        return str(x)

    def upload(self, x: int) -> str:
        return str(x)

class MyDownloadOnlyDataProvider(DataProviderDownload):
    name = "download-only"

    def download(self, x: int) -> str:
        return str(x)

class MyBadDataProvider(DataProvider):
    name = "bad"

    def download(self, x: str) -> str:
        return x

    def upload(self, x: int) -> str:
        return str(x)
    
def some_library_function(d: DataProviderDownload | DataProviderUpload) -> None:
    if isinstance(d, DataProviderUpload):
        d.upload(42)
    if isinstance(d, DataProviderDownload):
        d.download(42)


some_library_function(MyGoodDataProvider())
some_library_function(MyBadDataProvider())
some_library_function(MyDownloadOnlyDataProvider())
