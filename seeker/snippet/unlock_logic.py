#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073

from sqlalchemy import select
from models import ElementUnlockRequirement, PlayerInventory, CubeFace



async def unlock_element(db, player_id: int, element_id: int) -> bool:
    """Разблокирует элемент для игрока"""
    from models import CubeFace

    # Проверяем можно ли разблокировать (существующая логика)
    if not await can_unlock_element(db, player_id, element_id):
        return False

    # СПИСЫВАЕМ РЕСУРСЫ (существующая логика)
    await consume_unlock_resources(db, player_id, element_id)

    # НАЗНАЧАЕМ ЭЛЕМЕНТ НА 1 СВОБОДНУЮ ГРАНЬ
    free_face_result = await db.execute(
        select(CubeFace).where(
            CubeFace.player_id == player_id,
            CubeFace.element_id == None
        ).limit(1)
    )
    free_face = free_face_result.scalar_one_or_none()

    if free_face:
        free_face.element_id = element_id
        free_face.is_unlocked = True  # ⬅️ РАЗБЛОКИРУЕМ ГРАНЬ!
        print(f"✅ Assigned element {element_id} to free face")

    # РАЗБЛОКИРУЕМ ВСЕ ГРАНИ С ЭТИМ ЭЛЕМЕНТОМ
    result_faces = await db.execute(
        select(CubeFace).where(
            CubeFace.player_id == player_id,
            CubeFace.element_id == element_id
        )
    )
    faces_with_element = result_faces.scalars().all()

    for face in faces_with_element:
        face.is_unlocked = True  # ⬅️ РАЗБЛОКИРУЕМ ГРАНЬ!
        print(f"✅ Unlocked face for element {element_id}")

    await db.commit()
    return True


async def can_unlock_element(db, player_id: int, element_id: int) -> bool:
    """Проверяет достаточно ли ресурсов для разблокировки элемента"""

    # Получаем требования для разблокировки
    result = await db.execute(
        select(ElementUnlockRequirement).where(
            ElementUnlockRequirement.element_id == element_id
        )
    )
    requirements = result.scalars().all()

    # Если требований нет - можно разблокировать бесплатно
    if not requirements:
        return True

    # Проверяем каждый requirement
    for req in requirements:
        # Проверяем есть ли нужное количество в инвентаре
        inventory_result = await db.execute(
            select(PlayerInventory).where(
                PlayerInventory.player_id == player_id,
                PlayerInventory.element_id == req.required_element_id
            )
        )
        inventory = inventory_result.scalar_one_or_none()

        # Если ресурса нет или недостаточно - нельзя разблокировать
        if not inventory or inventory.amount < req.required_amount:
            return False

    return True


async def consume_unlock_resources(db, player_id: int, element_id: int):
    """Списывает ресурсы за разблокировку элемента"""

    # Получаем требования
    result = await db.execute(
        select(ElementUnlockRequirement).where(
            ElementUnlockRequirement.element_id == element_id
        )
    )
    requirements = result.scalars().all()

    # Списываем каждый ресурс
    for req in requirements:
        inventory_result = await db.execute(
            select(PlayerInventory).where(
                PlayerInventory.player_id == player_id,
                PlayerInventory.element_id == req.required_element_id
            )
        )
        inventory = inventory_result.scalar_one_or_none()

        if inventory:
            inventory.amount -= req.required_amount
            # Если количество стало 0 или меньше, можно удалить запись
            if inventory.amount <= 0:
                await db.delete(inventory)