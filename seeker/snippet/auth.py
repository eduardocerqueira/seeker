#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073

from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from database import get_db
from models import Player, PlayerInventory  # ✅ УБРАЛ CubeFace

security = HTTPBearer()


async def get_player_by_device(device_id: str, db: AsyncSession):
    result = await db.execute(select(Player).where(Player.device_id == device_id))
    return result.scalar_one_or_none()


async def create_player(device_id: str, username: str, db: AsyncSession):
    auth_token = "**********"

    player = Player(
        username=username,
        device_id=device_id,
         "**********"= "**********"
    )

    db.add(player)
    await db.commit()
    await db.refresh(player)

    # Инициализируем элементы для нового игрока
    await initialize_player_elements(player.id, db)

    return player


async def get_player_by_token(credentials: "**********"
                              db: AsyncSession = Depends(get_db)):
    token = "**********"
    result = "**********"== token))
    player = result.scalar_one_or_none()

    if not player:
        raise HTTPException(status_code= "**********"="Invalid token")

    return player


async def initialize_player_elements(player_id: int, db: AsyncSession):
    """Инициализация начальных элементов и склада для нового игрока"""

    # ДОБАВЛЯЕМ НАЧАЛЬНЫЕ РЕСУРСЫ В ИНВЕНТАРЬ
    initial_elements = [
        {"element_id": 1, "amount": 50.0},  # Hydrogen
        {"element_id": 2, "amount": 50.0},  # Helium
    ]

    for element_data in initial_elements:
        inventory = PlayerInventory(
            player_id=player_id,
            element_id=element_data["element_id"],
            amount=element_data["amount"]
        )
        db.add(inventory)

    # ✅ ТЕПЕРЬ ГРАНИ ИНИЦИАЛИЗИРУЮТСЯ АВТОМАТИЧЕСКИ ПРИ СОЗДАНИИ CubeState
    # в функции update_cube_state_batch в endpoints.py

    await db.commit()
    print(f"✅ Initialized player {player_id} with starting resources")wait db.commit()
    print(f"✅ Initialized player {player_id} with starting resources")