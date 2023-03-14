#date: 2023-03-14T16:53:23Z
#url: https://api.github.com/gists/6591135be96a828b1b2b4b25134e065e
#owner: https://api.github.com/users/chaacxib

import abc
import httpx
import pydantic
import typing


class Settings(pydantic.BaseSettings):
  is_dev_environment: bool = True


class AbstractProviderAdapter(abc.ABC):
  @abc.abstractmethod
  def get_response() -> typing.Dict:
    ...


class ProviderAdapter(AbstractProviderAdapter):
  def get_response() -> typing.Dict:
    r = httpx.get('https://httpbin.org/get')
    return r.json()


class ProviderAdapterMock(AbstractProviderAdapter):
  def get_response() -> typing.Dict:
    return {"TEST": True}
    
    
if __name__ == "__main__":
  _SETTINGS = Settings()
  my_adapter: AbstractProviderAdapter = (
    ProviderAdapterMock
    if _SETTINGS.is_dev_environment else
    ProviderAdapter
  )
  result = my_adapter.get_response()
    

