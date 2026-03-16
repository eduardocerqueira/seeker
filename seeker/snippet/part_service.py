#date: 2026-03-16T17:42:17Z
#url: https://api.github.com/gists/cd0f68e1b665833ec1fa02e2742787a0
#owner: https://api.github.com/users/96tm

import datetime
from typing import List

from loguru import logger
from nanoid import generate
from sqlalchemy import delete, func, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.db.session import db_session_context
from app.models.part import PartModel
from app.schema.datatable import DataTableRequest, PartDataTableResponse
from app.schema.part import PartCreateSchema, PartUpdateSchema

async def create_part(db: AsyncSession, details: PartCreateSchema) -> PartModel:
    """Creates a new Part and saves to db

    Args:
        details (PartCreate): details of Part to create

    Returns:
        part (PartPublic): the newly created Part
    """

    part = PartModel()

    part.id = generate(_alphabet, _size)

    if details.name is None:
        details.name = part.id
    part.name = details.name
    part.description = details.description
    part.notes = details.notes
    part.footprint = details.footprint
    part.manufacturer = details.manufacturer
    part.mpn = details.mpn

    # TODO: use contextvar for db dependencies when https://github.com/pytest-dev/pytest-asyncio/pull/161 is merged
    # db_session: AsyncSession = db_session_context.get()
    db_session: AsyncSession = db
    async with db_session as session:
        session.add(part)
        try:
            await session.commit()

        except IntegrityError as ex:
            await session.rollback()
            logger.error("Part ID already exists in the database", ex)

    return part