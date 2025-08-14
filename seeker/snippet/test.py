#date: 2025-08-14T16:59:53Z
#url: https://api.github.com/gists/056e01ade89e0ccdb795b6c9d74c6f00
#owner: https://api.github.com/users/PlzTrustMe

import uuid
from unittest.mock import create_autospec

import pytest
from faker.proxy import Faker

from exploding_kittens.application.commands.lobby import (
    SetupLobbyCommand,
    SetupLobbyCommandHandler,
)
from exploding_kittens.application.common.errors.lobby import (
    UserAlreadyInLobbyError,
)
from exploding_kittens.application.common.gateways.lobby_gateway import (
    LobbyGateway,
)
from exploding_kittens.application.common.id_generator import IDGenerator
from exploding_kittens.application.common.identity_provider import (
    IdentityProvider,
)
from exploding_kittens.domain.lobby.lobby import Lobby, LobbyUser
from exploding_kittens.domain.lobby.lobby_id import LobbyID
from exploding_kittens.domain.shared.user_id import UserID


@pytest.fixture
def make_create_lobby_handler():
    def _factory(
        current_user_id: int, exists: bool, generated_lobby_id: uuid.UUID
    ) -> SetupLobbyCommandHandler:
        identity_provider = create_autospec(IdentityProvider, instance=True)
        lobby_gateway = create_autospec(LobbyGateway, instance=True)
        id_generator = create_autospec(IDGenerator, instance=True)

        identity_provider.get_current_user_id.return_value = current_user_id
        lobby_gateway.exists.return_value = exists
        id_generator.generate_lobby_id.return_value = generated_lobby_id

        return SetupLobbyCommandHandler(
            identity_provider=identity_provider,
            lobby_gateway=lobby_gateway,
            id_generator=id_generator,
        )

    return _factory


@pytest.mark.asyncio
async def test_successful_setup_lobby(
    make_create_lobby_handler, faker: Faker
) -> None:
    current_user_id = UserID(faker.random_int())
    exists = False
    lobby_id = LobbyID(faker.uuid4(cast_to=None))
    max_players_in_lobby = faker.random_int(min=2, max=4)
    command = SetupLobbyCommand(max_players_in_lobby=max_players_in_lobby)
    handler = make_create_lobby_handler(
        current_user_id=current_user_id,
        exists=exists,
        generated_lobby_id=lobby_id,
    )

    result = await handler.handle(command)

    assert result.lobby_id == lobby_id
    # noinspection PyUnresolvedReferences
    handler._lobby_gateway.save.assert_awaited_once_with(
        Lobby(
            object_id=lobby_id,
            owner_id=current_user_id,
            lobby_users=[LobbyUser(object_id=current_user_id)],
            max_players=max_players_in_lobby,
        )
    )


@pytest.mark.asyncio
async def test_cant_setup_lobby_if_user_already_in_lobby(
    make_create_lobby_handler, faker: Faker
) -> None:
    command = SetupLobbyCommand(max_players_in_lobby=2)
    handler = make_create_lobby_handler(
        current_user_id=1,
        exists=True,
        generated_lobby_id=LobbyID(faker.uuid4(cast_to=None)),
    )

    with pytest.raises(UserAlreadyInLobbyError):
        await handler.handle(command)

    # noinspection PyUnresolvedReferences
    handler._lobby_gateway.save.assert_not_awaited()