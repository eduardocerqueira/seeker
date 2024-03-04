#date: 2024-03-04T17:07:05Z
#url: https://api.github.com/gists/241402b97fa9d9daf1ee0fb2d26322fb
#owner: https://api.github.com/users/Sehat1137

import asyncio
from dataclasses import dataclass
from typing import Protocol, Annotated

import uvicorn

from dishka import Provider, Scope, provide
from dishka import make_async_container
from dishka.integrations import faststream as faststream_integration
from dishka.integrations import litestar as litestar_integration
from dishka.integrations.base import Depends
from dishka.integrations.faststream import inject as faststream_inject
from dishka.integrations.litestar import inject as litestar_inject

from faststream import FastStream
from faststream.rabbit import RabbitBroker, RabbitRouter, RabbitRoute

from litestar import Litestar, route, HttpMethod
from litestar.dto import DTOConfig, DataclassDTO

from pydantic import Field
from pydantic_settings import BaseSettings


# config

class RabbitMQSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5672
    login: str = "user"
    password: "**********"


class Config(BaseSettings):
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)


# entities

@dataclass(slots=True)
class UserDM:
    id: int
    name: str


# interactor

class IUserInfo(Protocol):
    async def get_user(self, user_id: int) -> UserDM:
        raise NotImplementedError


class Interactor:
    def __init__(self, user_info: IUserInfo) -> None:
        self._user_info = user_info

    async def __call__(self, user_id: int) -> UserDM:
        user_dm = await self._user_info.get_user(user_id=user_id)
        return user_dm


# repository


class UserRepo:
    def __init__(self) -> None:
        pass

    async def get_user(self, user_id: int) -> UserDM:
        return UserDM(id=user_id, name="John Doe")


# handlers


class UserSchemaHTTP(DataclassDTO[UserDM]):
    config = DTOConfig(partial=True)


@route(http_method=HttpMethod.GET, dto=UserSchemaHTTP, path="/user")
@litestar_inject
async def get_user_http(user_id: int, interactor: Annotated[Interactor, Depends()]) -> UserDM:
    user_dm = await interactor(user_id=user_id)
    return user_dm


@faststream_inject
async def get_user_amqp(
        user_id: int,
        interactor: Annotated[Interactor, Depends()],
        broker: Annotated[RabbitBroker, Depends()]
) -> None:
    user_dm = await interactor(user_id=user_id)
    await broker.publish(
        {'id': user_dm.id, 'name': user_dm.name},
        queue="response"
    )


# IOC


class SomeProvider(Provider):
    @provide(scope=Scope.APP)
    def get_configs(self) -> Config:
        return Config()

    @provide(scope=Scope.APP)
    def broker(self, config: Config) -> RabbitBroker:
        broker = RabbitBroker(
            host=config.rabbitmq.host,
            port=config.rabbitmq.port,
            login=config.rabbitmq.login,
            password= "**********"
            virtualhost="/",
        )
        return broker

    @provide(scope=Scope.REQUEST)
    def user_info(self) -> IUserInfo:
        return UserRepo()

    interactor = provide(Interactor, scope=Scope.REQUEST)


async def create_app():
    container = make_async_container(SomeProvider())

    broker = await container.get(RabbitBroker)
    amqp_routes = RabbitRouter(
        handlers=(
            RabbitRoute(get_user_amqp, "request"),
        )
    )
    broker.include_router(amqp_routes)
    faststream_integration.setup_dishka(container, FastStream(broker))

    http_routes = [
        get_user_http
    ]
    http = Litestar(
        route_handlers=http_routes,
        on_startup=[broker.start],
        on_shutdown=[broker.close],
    )
    litestar_integration.setup_dishka(container, http)
    return http


if __name__ == "__main__":
    app = asyncio.run(create_app())
    uvicorn.run(app, host="0.0.0.0", port=8000)
.0", port=8000)
