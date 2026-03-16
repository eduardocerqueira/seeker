#date: 2026-03-16T17:42:17Z
#url: https://api.github.com/gists/cd0f68e1b665833ec1fa02e2742787a0
#owner: https://api.github.com/users/96tm

import pytest
from loguru import logger  # noqa:
from sqlalchemy.ext.asyncio import AsyncSession  # noqa:

from app.models.part import PartModel
from app.services import part_service


class TestPartService:
    @pytest.mark.asyncio
    async def test_add_single_part(
        self,
        part_create_schema_factory,
        db_session,
    ):

        obj_in = part_create_schema_factory

        item: PartModel = await part_service.create_part(db_session, obj_in)  # noqa:

        assert item.name == obj_in.name
        assert item.id is not None
        assert len(item.id) == 26

    @pytest.mark.asyncio
    async def test_part_count_with_empty_db(self, db_session):

        assert await part_service.get_part_count(db_session) == 0