#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073

import threading
import time
from typing import List, Dict
from sqlalchemy import select
import math
import numpy as np
from math import cos, sin, radians

# Глобальные переменные для батчинга
cube_updates_batch = []
harvest_batch = []
batch_lock = threading.Lock()


def calculate_active_faces(cube_position_id: int, rotation_x: float, rotation_y: float, rotation_z: float):
    """
    Определяет какие грани кубика смотрят наружу
    С УЧЕТОМ ТЕКУЩЕЙ ПОЗИЦИИ ПОСЛЕ ВРАЩЕНИЯ
    """
    # Вычисляем начальные координаты из ID
    initial_x = (cube_position_id // 100) - 1
    initial_y = ((cube_position_id % 100) // 10) - 1
    initial_z = (cube_position_id % 10) - 1

    # ВЫЧИСЛЯЕМ ТЕКУЩУЮ ПОЗИЦИЮ после вращения
    current_x, current_y, current_z = calculate_current_position(
        initial_x, initial_y, initial_z, rotation_x, rotation_y, rotation_z
    )

    # Преобразуем углы Эйлера в радианы
    rx = math.radians(rotation_x)
    ry = math.radians(rotation_y)
    rz = math.radians(rotation_z)

    # Нормальные векторы для каждой грани кубика (в локальных координатах)
    face_normals = {
        'front': (0, 0, 1),
        'back': (0, 0, -1),
        'left': (-1, 0, 0),
        'right': (1, 0, 0),
        'top': (0, 1, 0),
        'bottom': (0, -1, 0)
    }

    # Поворачиваем нормальные векторы согласно вращению кубика
    active_faces = []

    for face_name, normal in face_normals.items():
        # Поворачиваем вектор нормали
        rotated_normal = rotate_vector(normal, rx, ry, rz)

        # ✅ ПЕРЕДАЕМ ТЕКУЩИЕ ПОЗИЦИИ В is_face_outward
        if is_face_outward(current_x, current_y, current_z, rotated_normal):
            active_faces.append(face_name)

    return active_faces


def rotate_vector(vector, rx, ry, rz):
    """Поворачивает вектор на углы Эйлера"""
    x, y, z = vector

    # Поворот вокруг X
    y1 = y * math.cos(rx) - z * math.sin(rx)
    z1 = y * math.sin(rx) + z * math.cos(rx)

    # Поворот вокруг Y
    x2 = x * math.cos(ry) + z1 * math.sin(ry)
    z2 = -x * math.sin(ry) + z1 * math.cos(ry)

    # Поворот вокруг Z
    x3 = x2 * math.cos(rz) - y1 * math.sin(rz)
    y3 = x2 * math.sin(rz) + y1 * math.cos(rz)

    return (x3, y3, z2)

def is_face_outward(current_x: int, current_y: int, current_z: int, normal):
    """
    Определяет смотрит ли грань наружу большого куба
    ИСПОЛЬЗУЕТ ТЕКУЩИЕ ПОЗИЦИИ куба после вращения
    """
    nx, ny, nz = normal

    # ✅ ИСПОЛЬЗУЕМ ТЕКУЩИЕ ПОЗИЦИИ вместо cube_position_id!
    if nx > 0.7 and current_x == 1: return True  # Правая грань на правой стороне
    if nx < -0.7 and current_x == -1: return True  # Левая грань на левой стороне
    if ny > 0.7 and current_y == 1: return True  # Верхняя грань на верхней стороне
    if ny < -0.7 and current_y == -1: return True  # Нижняя грань на нижней стороне
    if nz > 0.7 and current_z == 1: return True  # Передняя грань на передней стороне
    if nz < -0.7 and current_z == -1: return True  # Задняя грань на задней стороне

    return False


def determine_default_element(cube_position_id: int, face_direction: str) -> int:
    """Назначает элементы граням при создании - ЭЛЕМЕНТЫ ПРИВЯЗАНЫ К ГРАНЯМ НАВСЕГДА"""
    import hashlib

    # ✅ СОЗДАЕМ УНИКАЛЬНЫЙ ХЭШ ДЛЯ КАЖДОЙ ГРАНИ
    # Элемент будет ПРИКРЕПЛЕН к этой грани навсегда
    seed_string = f"{cube_position_id}_{face_direction}"
    seed_value = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

    # ✅ РАСПРЕДЕЛЯЕМ 118 ЭЛЕМЕНТОВ ПО 156 ГРАНЯМ
    # Каждая грань получает ПОСТОЯННЫЙ элемент
    element_id = (seed_value % 118) + 1

    return element_id


def calculate_harvest(player_id: int):
    """Рассчитывает добычу на основе активных граней и сохраняет НА ГРАНИ CubeState"""
    try:
        from database import SyncSessionLocal
        from models import CubeState

        with SyncSessionLocal() as db:
            # ✅ ПОЛУЧАЕМ ВСЕ КУБЫ ИГРОКА С АКТИВНЫМИ ГРАНЯМИ
            all_cubes = db.execute(
                select(CubeState).where(CubeState.player_id == player_id)
            ).scalars().all()

            print(f"🔍 Player {player_id}: {len(all_cubes)} cubes total")

            active_faces_count = 0

            for cube in all_cubes:
                # ✅ ПРОВЕРЯЕМ КАЖДУЮ ГРАНЬ КУБА
                faces_to_process = [
                    ('front', cube.front_element_id, cube.front_is_active, 'front_current_amount'),
                    ('back', cube.back_element_id, cube.back_is_active, 'back_current_amount'),
                    ('left', cube.left_element_id, cube.left_is_active, 'left_current_amount'),
                    ('right', cube.right_element_id, cube.right_is_active, 'right_current_amount'),
                    ('top', cube.top_element_id, cube.top_is_active, 'top_current_amount'),
                    ('bottom', cube.bottom_element_id, cube.bottom_is_active, 'bottom_current_amount'),
                ]

                for face_name, element_id, is_active, amount_field in faces_to_process:
                    if is_active and element_id:
                        active_faces_count += 1
                        production_rate = get_element_production_rate(element_id)

                        # ✅ ОБНОВЛЯЕМ КОЛИЧЕСТВО НА ГРАНИ
                        current_amount = getattr(cube, amount_field) or 0
                        storage_capacity = get_max_storage(element_id, 1)
                        new_amount = min(current_amount + production_rate, storage_capacity)
                        setattr(cube, amount_field, new_amount)

                        print(
                            f"🔍 Cube {cube.cube_position_id}-{face_name}: producing {production_rate}, new amount: {new_amount}")

            db.commit()

            if active_faces_count > 0:
                print(f"✅ Produced resources on {active_faces_count} active faces for player {player_id}")
            else:
                print(f"⏳ No active faces for player {player_id}")

    except Exception as e:
        print(f"❌ Error calculating harvest: {e}")

def get_element_production_rate(element_id: int) -> float:
    """Возвращает скорость добычи для элемента из БД"""
    # Временно заглушка - нужно получать из БД
    rates = {
        1: 0.15,  # Hydrogen
        2: 0.12,  # Helium
        3: 0.18,  # Lithium
    }
    return rates.get(element_id, 0.1)

def get_max_storage(element_id: int, storage_level: int) -> float:
    """Возвращает максимальную вместимость"""
    base_capacity = {
        1: 120,  # Hydrogen
        2: 100,  # Helium
        3: 150,  # Lithium
    }
    return base_capacity.get(element_id, 100) * storage_level

async def collect_resources_from_face(db, player_id: int, cube_position_id: int, face_direction: str):
    """Собирает ресурсы с конкретной грани на склад игрока"""
    from models import CubeFace, PlayerInventory

    # Находим грань
    face_result = await db.execute(
        select(CubeFace)
        .where(
            CubeFace.player_id == player_id,
            CubeFace.cube_position_id == cube_position_id,
            CubeFace.face_direction == face_direction
        )
    )
    face = face_result.scalar_one_or_none()

    # ИСПРАВЛЯЕМ НАЗВАНИЕ ПОЛЯ
    current_amount = getattr(face, 'current_amount', 0) or 0

    if face and face.element_id and current_amount > 0:
        collected_amount = current_amount

        # Находим или создаем запись в инвентаре
        inventory_result = await db.execute(
            select(PlayerInventory)
            .where(
                PlayerInventory.player_id == player_id,
                PlayerInventory.element_id == face.element_id
            )
        )
        inventory = inventory_result.scalar_one_or_none()

        if inventory:
            inventory.amount += collected_amount
        else:
            inventory = PlayerInventory(
                player_id=player_id,
                element_id=face.element_id,
                amount=collected_amount
            )
            db.add(inventory)

        # Обнуляем хранилище на грани
        face.current_amount = 0  # ИСПРАВЛЕНО
        await db.commit()

        return collected_amount
    return 0
async def produce_resources_for_player(db, player_id: int):
    """Производит ресурсы на всех активных гранях игрока"""
    from models import CubeFace

    # Получаем все активные грани игрока
    faces_result = await db.execute(
        select(CubeFace)
        .where(
            CubeFace.player_id == player_id,
            CubeFace.is_active == True,
            CubeFace.element_id.isnot(None)
        )
    )
    active_faces = faces_result.scalars().all()

    for face in active_faces:
        production_rate = get_element_production_rate(face.element_id)

        # Увеличиваем количество на грани
        if face.stored_amount is None:
            face.stored_amount = 0

        # Получаем лимит хранения грани
        storage_capacity = get_max_storage(face.element_id, 1)  # Пока уровень 1

        # Производим, но не превышаем лимит
        new_amount = min(face.stored_amount + production_rate, storage_capacity)
        face.stored_amount = new_amount

    await db.commit()

def process_batches_sync():
    """Синхронная обработка пакетов"""
    from database import SyncSessionLocal

    with batch_lock:
        # Обработка кубиков
        if cube_updates_batch:
            batch = cube_updates_batch.copy()
            cube_updates_batch.clear()
            process_cube_batch_sync(batch)

        # Обработка ресурсов (авто-добыча для всех игроков)
        calculate_harvest_for_all_players()

def calculate_harvest_for_all_players():
    """Производит ресурсы для всех игроков"""
    try:
        from database import SyncSessionLocal
        from models import Player

        with SyncSessionLocal() as db:
            players = db.execute(select(Player)).scalars().all()

            for player in players:
                # Автоматическое производство на активных гранях
                calculate_harvest(player.id)

            print(f"✅ Авто-производство завершено для {len(players)} игроков")

    except Exception as e:
        print(f"❌ Ошибка авто-производства: {e}")

def start_background_processor():
    """Запуск фоновой обработки пакетов"""

    def processor():
        while True:
            time.sleep(10)  # Обработка каждые 10 секунд
            process_batches_sync()

    thread = threading.Thread(target=processor, daemon=True)
    thread.start()
    print("✅ Фоновый процессор запущен (с определением активных граней)")


def calculate_current_position(initial_x, initial_y, initial_z, rotation_x, rotation_y, rotation_z):
    """
    Вычисляет текущую позицию куба на основе начальной позиции и вращений
    """
    # Преобразуем углы в радианы
    rx = math.radians(rotation_x)
    ry = math.radians(rotation_y)
    rz = math.radians(rotation_z)

    # Начальная позиция как вектор
    initial_pos = np.array([initial_x, initial_y, initial_z])

    # Матрицы вращения
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])

    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])

    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])

    # Общая матрица вращения
    R = Rz @ Ry @ Rx

    # Применяем вращение к позиции
    current_pos = R @ initial_pos

    # Округляем до целых (кубы остаются в узлах сетки)
    return (
        round(current_pos[0]),
        round(current_pos[1]),
        round(current_pos[2])
    )