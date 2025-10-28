#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from auth import get_player_by_token
from database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from rotate import calculate_active_faces, calculate_current_position, determine_default_element

router = APIRouter()


# Модели запросов
class CubeStateData(BaseModel):
    cube_position_id: int
    rotation_x: float
    rotation_y: float
    rotation_z: float


class DeviceAuthRequest(BaseModel):
    device_id: str
    username: str = None


class HarvestData(BaseModel):
    element_id: int
    amount: float


@router.post("/auth/device-login")
async def device_login(auth_data: DeviceAuthRequest, db: AsyncSession = Depends(get_db)):
    from auth import get_player_by_device, create_player

    player = await get_player_by_device(auth_data.device_id, db)

    if not player:
        username = auth_data.username or f"Player_{auth_data.device_id[-6:]}"
        player = await create_player(auth_data.device_id, username, db)
        is_new_player = True
    else:
        is_new_player = False

    return {
        "auth_token": "**********"
        "player_id": player.id,
        "username": player.username,
        "is_new_player": is_new_player
    }


@router.get("/player/state")
async def get_player_state(
    player = "**********"
    db: AsyncSession = Depends(get_db)
):
    """Получить состояние игрока (кубы, инвентарь)"""
    from models import Player, CubeState

    # Получаем состояния кубиков
    result = await db.execute(
        select(CubeState).where(CubeState.player_id == player.id)
    )
    cube_states = result.scalars().all()

    cube_states_data = []
    for cube in cube_states:
        cube_states_data.append({
            "cube_position_id": cube.cube_position_id,
            "rotation_x": cube.rotation_x,
            "rotation_y": cube.rotation_y,
            "rotation_z": cube.rotation_z,
        })

    return {
        "player": {
            "id": player.id,
            "username": player.username
        },
        "cube_states": cube_states_data
    }


@router.post("/player/cube-state/batch")
async def update_cube_state_batch(
        cubes: List[CubeStateData],
        player= "**********"
        db: AsyncSession = Depends(get_db)
):
    try:
        from models import CubeState

        player_id = player.id

        for cube in cubes:
            # Рассчитываем текущие позиции
            initial_x = (cube.cube_position_id // 100) - 1
            initial_y = ((cube.cube_position_id % 100) // 10) - 1
            initial_z = (cube.cube_position_id % 10) - 1

            current_x, current_y, current_z = calculate_current_position(
                initial_x, initial_y, initial_z,
                cube.rotation_x, cube.rotation_y, cube.rotation_z
            )

            # Дебаг-вывод
            print(f"Cube {cube.cube_position_id}:")
            print(f"   initial: ({initial_x}, {initial_y}, {initial_z})")
            print(f"   rotation: ({cube.rotation_x}, {cube.rotation_y}, {cube.rotation_z})")
            print(f"   current: ({current_x}, {current_y}, {current_z})")

            # Найти или создать запись CubeState
            cube_state_result = await db.execute(
                select(CubeState).where(
                    CubeState.player_id == player_id,
                    CubeState.cube_position_id == cube.cube_position_id
                )
            )
            cube_state = cube_state_result.scalar_one_or_none()

            # Определяем активные грани
            active_faces = calculate_active_faces(
                cube.cube_position_id, cube.rotation_x, cube.rotation_y, cube.rotation_z
            )

            if cube_state:
                # Обновляем существующий куб
                cube_state.rotation_x = cube.rotation_x
                cube_state.rotation_y = cube.rotation_y
                cube_state.rotation_z = cube.rotation_z
                cube_state.initial_x = current_x
                cube_state.initial_y = current_y
                cube_state.initial_z = current_z

                # Обновляем активность граней
                cube_state.front_is_active = 'front' in active_faces and cube_state.front_is_unlocked
                cube_state.back_is_active = 'back' in active_faces and cube_state.back_is_unlocked
                cube_state.left_is_active = 'left' in active_faces and cube_state.left_is_unlocked
                cube_state.right_is_active = 'right' in active_faces and cube_state.right_is_unlocked
                cube_state.top_is_active = 'top' in active_faces and cube_state.top_is_unlocked
                cube_state.bottom_is_active = 'bottom' in active_faces and cube_state.bottom_is_unlocked

                print(f"✅ Updated existing cube {cube.cube_position_id}")
            else:
                # Создаем новый куб с начальными значениями граней
                default_front = determine_default_element(cube.cube_position_id, 'front')
                default_back = determine_default_element(cube.cube_position_id, 'back')
                default_left = determine_default_element(cube.cube_position_id, 'left')
                default_right = determine_default_element(cube.cube_position_id, 'right')
                default_top = determine_default_element(cube.cube_position_id, 'top')
                default_bottom = determine_default_element(cube.cube_position_id, 'bottom')

                cube_state = CubeState(
                    player_id=player_id,
                    cube_position_id=cube.cube_position_id,
                    rotation_x=cube.rotation_x,
                    rotation_y=cube.rotation_y,
                    rotation_z=cube.rotation_z,
                    initial_x=current_x,
                    initial_y=current_y,
                    initial_z=current_z,

                    # Инициализируем грани
                    front_element_id=default_front,
                    front_is_unlocked=default_front in [1, 2],
                    front_is_active='front' in active_faces and (default_front in [1, 2]),

                    back_element_id=default_back,
                    back_is_unlocked=default_back in [1, 2],
                    back_is_active='back' in active_faces and (default_back in [1, 2]),

                    left_element_id=default_left,
                    left_is_unlocked=default_left in [1, 2],
                    left_is_active='left' in active_faces and (default_left in [1, 2]),

                    right_element_id=default_right,
                    right_is_unlocked=default_right in [1, 2],
                    right_is_active='right' in active_faces and (default_right in [1, 2]),

                    top_element_id=default_top,
                    top_is_unlocked=default_top in [1, 2],
                    top_is_active='top' in active_faces and (default_top in [1, 2]),

                    bottom_element_id=default_bottom,
                    bottom_is_unlocked=default_bottom in [1, 2],
                    bottom_is_active='bottom' in active_faces and (default_bottom in [1, 2]),
                )
                db.add(cube_state)
                print(f"✅ Created NEW cube {cube.cube_position_id}")

        await db.commit()
        print(f"✅ Saved {len(cubes)} cubes for player {player_id}")
        return {"message": f"Batch updated {len(cubes)} cubes", "success": True}

    except Exception as e:
        await db.rollback()
        print(f"❌ Error in batch update: {e}")
        return {"message": f"Error: {str(e)}", "success": False}


@router.post("/player/cube-state/single")
async def update_cube_state_single(
        cube: CubeStateData,
        player= "**********"
        db: AsyncSession = Depends(get_db)
):
    """Сохранение одного куба"""
    try:
        from models import CubeState
        from rotate import calculate_active_faces, calculate_current_position

        player_id = player.id

        # Используем ту же логику что и для batch
        cubes = [cube]
        return await update_cube_state_batch(cubes, player, db)

    except Exception as e:
        return {"message": f"Error: {str(e)}", "success": False}


@router.get("/player/active-faces")
async def get_active_faces(
    player = "**********"
    db: AsyncSession = Depends(get_db)
):
    """Получить ВСЕ активные грани игрока"""
    from models import CubeState, Element
    from sqlalchemy import select

    result = await db.execute(
        select(CubeState).where(CubeState.player_id == player.id)
    )
    all_cubes = result.scalars().all()

    print(f"🔍 Active faces check: found {len(all_cubes)} cubes for player {player.id}")

    active_faces_list = []
    total_active = 0

    for cube in all_cubes:
        # Проверяем каждую грань куба
        faces_to_check = [
            ('front', cube.front_element_id, cube.front_is_active, cube.front_is_unlocked),
            ('back', cube.back_element_id, cube.back_is_active, cube.back_is_unlocked),
            ('left', cube.left_element_id, cube.left_is_active, cube.left_is_unlocked),
            ('right', cube.right_element_id, cube.right_is_active, cube.right_is_unlocked),
            ('top', cube.top_element_id, cube.top_is_active, cube.top_is_unlocked),
            ('bottom', cube.bottom_element_id, cube.bottom_is_active, cube.bottom_is_unlocked),
        ]

        for face_direction, element_id, is_active, is_unlocked in faces_to_check:
            if is_active and element_id and is_unlocked:
                # Получаем информацию об элементе
                element_result = await db.execute(
                    select(Element).where(Element.id == element_id)
                )
                element = element_result.scalar_one_or_none()

                if element:
                    active_faces_list.append({
                        "cube_id": cube.cube_position_id,
                        "face_direction": face_direction,
                        "element": element.symbol,
                        "element_name": element.name,
                        "element_id": element_id,
                        "is_active": is_active
                    })
                    total_active += 1
                    print(f"🔍 Active face: cube {cube.cube_position_id} {face_direction} - {element.symbol}")

    print(f"✅ Total active faces: {total_active}")
    return {
        "active_faces": active_faces_list,
        "total_active": total_active
    }

# Остальные эндпоинты (inventory, unlock, collect) остаются без изменений
@router.get("/player/inventory")
async def get_player_inventory(
        player= "**********"
        db: AsyncSession = Depends(get_db)
):
    """Получает склад игрока"""
    from models import PlayerInventory, Element
    from sqlalchemy import select

    result = await db.execute(
        select(PlayerInventory, Element)
        .join(Element, PlayerInventory.element_id == Element.id)
        .where(PlayerInventory.player_id == player.id)
    )

    inventory = []
    for inv_item, element in result:
        inventory.append({
            "element_id": element.id,
            "symbol": element.symbol,
            "name": element.name,
            "amount": inv_item.amount
        })

    return {"inventory": inventory}


@router.post("/player/collect-face")
async def collect_face_resources(
        cube_position_id: int,
        face_direction: str,
        player= "**********"
        db: AsyncSession = Depends(get_db)
):
    """Собирает ресурсы с КОНКРЕТНОЙ грани"""
    from models import CubeState, PlayerInventory, Element
    from rotate import get_element_production_rate, get_max_storage
    from sqlalchemy import select

    # Находим куб и грань
    cube_result = await db.execute(
        select(CubeState).where(
            CubeState.player_id == player.id,
            CubeState.cube_position_id == cube_position_id
        )
    )
    cube = cube_result.scalar_one_or_none()

    if not cube:
        raise HTTPException(status_code=404, detail="Cube not found")

    # Получаем данные грани
    face_data = None
    if face_direction == 'front':
        face_data = (cube.front_element_id, cube.front_current_amount, 'front_current_amount')
    elif face_direction == 'back':
        face_data = (cube.back_element_id, cube.back_current_amount, 'back_current_amount')
    elif face_direction == 'left':
        face_data = (cube.left_element_id, cube.left_current_amount, 'left_current_amount')
    elif face_direction == 'right':
        face_data = (cube.right_element_id, cube.right_current_amount, 'right_current_amount')
    elif face_direction == 'top':
        face_data = (cube.top_element_id, cube.top_current_amount, 'top_current_amount')
    elif face_direction == 'bottom':
        face_data = (cube.bottom_element_id, cube.bottom_current_amount, 'bottom_current_amount')

    if not face_data or not face_data[0] or face_data[1] <= 0:
        return {
            "success": False,
            "message": "На этой грани нет ресурсов для сбора",
            "collected_amount": 0
        }

    element_id, current_amount, amount_field = face_data
    collected_amount = current_amount

    # Находим или создаем запись в инвентаре
    inventory_result = await db.execute(
        select(PlayerInventory).where(
            PlayerInventory.player_id == player.id,
            PlayerInventory.element_id == element_id
        )
    )
    inventory = inventory_result.scalar_one_or_none()

    if inventory:
        inventory.amount += collected_amount
    else:
        inventory = PlayerInventory(
            player_id=player.id,
            element_id=element_id,
            amount=collected_amount
        )
        db.add(inventory)

    # Обнуляем хранилище на грани
    setattr(cube, amount_field, 0)

    await db.commit()

    return {
        "success": True,
        "message": f"Собрано {collected_amount} ресурсов с грани",
        "collected_amount": collected_amount,
        "cube_position_id": cube_position_id,
        "face_direction": face_direction
    }ted_amount,
        "cube_position_id": cube_position_id,
        "face_direction": face_direction
    }