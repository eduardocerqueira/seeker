#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073

#!/usr/bin/env python3
"""
Скрипт для автоматического создания всех таблиц в базе данных
"""

import sys
import os

# Добавляем путь к текущей директории для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import sync_engine
from models import Base
from config import config


def create_tables():
    """Создает все таблицы в базе данных"""
    print("🔄 Создание таблиц в базе данных...")

    try:
        # Удаляем старые таблицы если существуют
        Base.metadata.drop_all(bind=sync_engine)
        print("✅ Старые таблицы удалены")

        # Создаем новые таблицы
        Base.metadata.create_all(bind=sync_engine)
        print("✅ Все таблицы успешно созданы!")

        # Проверяем созданные таблицы
        check_tables()

    except Exception as e:
        print(f"❌ Ошибка при создании таблиц: {e}")
        return False

    return True


def check_tables():
    """Проверяет что все таблицы созданы и имеют нужные колонки"""
    print("\n🔍 Проверка таблиц...")

    from sqlalchemy import inspect

    inspector = inspect(sync_engine)
    tables = inspector.get_table_names()

    required_tables = [
        'players', 'cube_states', 'elements',
        'player_inventory', 'element_unlock_requirements'
    ]

    for table in required_tables:
        if table in tables:
            print(f"✅ Таблица {table} создана")
            columns = inspector.get_columns(table)
            column_names = [col['name'] for col in columns]
            print(f"   Колонки ({len(column_names)}): {column_names}")
        else:
            print(f"❌ Таблица {table} отсутствует")


def add_initial_data():
    """Добавляет начальные данные (элементы периодической таблицы)"""
    print("\n🔄 Добавление начальные данные...")

    from models import Element, ElementUnlockRequirement
    from database import SyncSessionLocal

    with SyncSessionLocal() as db:
        # Проверяем есть ли уже элементы
        from sqlalchemy import select
        existing_elements = db.execute(select(Element)).scalars().all()

        if not existing_elements:
            print("✅ Добавление элементов периодической таблицы...")

            # Базовые элементы (H и He разблокированы изначально)
            elements_data = [
             {"id": 1, "symbol": "H", "name": "Hydrogen", "base_production_rate": 0.15, "base_storage_capacity": 120,
                "is_initial": True},
             {"id": 2, "symbol": "He", "name": "Helium", "base_production_rate": 0.12, "base_storage_capacity": 100,
                "is_initial": True},
             {"id": 3, "symbol": "Li", "name": "Lithium", "base_production_rate": 0.18, "base_storage_capacity": 150,
                "is_initial": False},
             {"id": 4, "symbol": "Be", "name": "Beryllium", "base_production_rate": 0.16,
                "base_storage_capacity": 140, "is_initial": False},
             {"id": 5, "symbol": "B", "name": "Boron", "base_production_rate": 0.14, "base_storage_capacity": 130,
                "is_initial": False},
             {"id": 6, "symbol": "C", "name": "Carbon", "base_production_rate": 0.20, "base_storage_capacity": 160,
                "is_initial": True},
             {"id": 7, "symbol": "N", "name": "Nitrogen", "base_production_rate": 0.17, "base_storage_capacity": 145,
                "is_initial": True},
             {"id": 8, "symbol": "O", "name": "Oxygen", "base_production_rate": 0.22, "base_storage_capacity": 180,
                "is_initial": True},
             {"id": 9, "symbol": "F", "name": "Fluorine", "base_production_rate": 0.13, "base_storage_capacity": 125,
                "is_initial": False},
             {"id": 10, "symbol": "Ne", "name": "Neon", "base_production_rate": 0.11, "base_storage_capacity": 110,
                 "is_initial": False},
             {"id": 11, "symbol": "Na", "name": "Sodium", "base_production_rate": 0.19, "base_storage_capacity": 155,
                 "is_initial": False},
             {"id": 12, "symbol": "Mg", "name": "Magnesium", "base_production_rate": 0.21,
                 "base_storage_capacity": 165, "is_initial": False},
             {"id": 13, "symbol": "Al", "name": "Aluminium", "base_production_rate": 0.23,
                 "base_storage_capacity": 175, "is_initial": False},
             {"id": 14, "symbol": "Si", "name": "Silicon", "base_production_rate": 0.25,
                 "base_storage_capacity": 190, "is_initial": False},
             {"id": 15, "symbol": "P", "name": "Phosphorus", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
             {"id": 16, "symbol": "S", "name": "Sulfur", "base_production_rate": 0.16, "base_storage_capacity": 140,
                 "is_initial": False},
             {"id": 17, "symbol": "Cl", "name": "Chlorine", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
             {"id": 18, "symbol": "Ar", "name": "Argon", "base_production_rate": 0.12, "base_storage_capacity": 120,
                 "is_initial": False},
             {"id": 19, "symbol": "K", "name": "Potassium", "base_production_rate": 0.20,
                 "base_storage_capacity": 160, "is_initial": False},
             {"id": 20, "symbol": "Ca", "name": "Calcium", "base_production_rate": 0.22,
                 "base_storage_capacity": 170, "is_initial": False},
             {"id": 21, "symbol": "Sc", "name": "Scandium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
             {"id": 22, "symbol": "Ti", "name": "Titanium", "base_production_rate": 0.24,
                 "base_storage_capacity": 185, "is_initial": False},
             {"id": 23, "symbol": "V", "name": "Vanadium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
             {"id": 24, "symbol": "Cr", "name": "Chromium", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
             {"id": 25, "symbol": "Mn", "name": "Manganese", "base_production_rate": 0.21,
                 "base_storage_capacity": 165, "is_initial": False},
             {"id": 26, "symbol": "Fe", "name": "Iron", "base_production_rate": 0.28, "base_storage_capacity": 220,
                 "is_initial": False},
             {"id": 27, "symbol": "Co", "name": "Cobalt", "base_production_rate": 0.16, "base_storage_capacity": 140,
                 "is_initial": False},
             {"id": 28, "symbol": "Ni", "name": "Nickel", "base_production_rate": 0.23, "base_storage_capacity": 180,
                 "is_initial": False},
             {"id": 29, "symbol": "Cu", "name": "Copper", "base_production_rate": 0.20, "base_storage_capacity": 160,
                 "is_initial": False},
             {"id": 30, "symbol": "Zn", "name": "Zinc", "base_production_rate": 0.18, "base_storage_capacity": 150,
                 "is_initial": False},
             {"id": 31, "symbol": "Ga", "name": "Gallium", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
             {"id": 32, "symbol": "Ge", "name": "Germanium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
             {"id": 33, "symbol": "As", "name": "Arsenic", "base_production_rate": 0.12,
                 "base_storage_capacity": 120, "is_initial": False},
             {"id": 34, "symbol": "Se", "name": "Selenium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
             {"id": 35, "symbol": "Br", "name": "Bromine", "base_production_rate": 0.11,
                 "base_storage_capacity": 115, "is_initial": False},
             {"id": 36, "symbol": "Kr", "name": "Krypton", "base_production_rate": 0.10,
                 "base_storage_capacity": 110, "is_initial": False},
             {"id": 37, "symbol": "Rb", "name": "Rubidium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
             {"id": 38, "symbol": "Sr", "name": "Strontium", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
             {"id": 39, "symbol": "Y", "name": "Yttrium", "base_production_rate": 0.16, "base_storage_capacity": 140,
                 "is_initial": False},
             {"id": 40, "symbol": "Zr", "name": "Zirconium", "base_production_rate": 0.22,
                 "base_storage_capacity": 170, "is_initial": False},
             {"id": 41, "symbol": "Nb", "name": "Niobium", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
             {"id": 42, "symbol": "Mo", "name": "Molybdenum", "base_production_rate": 0.24,
                 "base_storage_capacity": 185, "is_initial": False},
             {"id": 43, "symbol": "Tc", "name": "Technetium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
             {"id": 44, "symbol": "Ru", "name": "Ruthenium", "base_production_rate": 0.20,
                 "base_storage_capacity": 160, "is_initial": False},
             {"id": 45, "symbol": "Rh", "name": "Rhodium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
             {"id": 46, "symbol": "Pd", "name": "Palladium", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
             {"id": 47, "symbol": "Ag", "name": "Silver", "base_production_rate": 0.17, "base_storage_capacity": 145,
                 "is_initial": False},
             {"id": 48, "symbol": "Cd", "name": "Cadmium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
             {"id": 49, "symbol": "In", "name": "Indium", "base_production_rate": 0.14, "base_storage_capacity": 130,
                 "is_initial": False},
             {"id": 50, "symbol": "Sn", "name": "Tin", "base_production_rate": 0.21, "base_storage_capacity": 165,
                 "is_initial": False},
             {"id": 51, "symbol": "Sb", "name": "Antimony", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
             {"id": 52, "symbol": "Te", "name": "Tellurium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
             {"id": 53, "symbol": "I", "name": "Iodine", "base_production_rate": 0.12, "base_storage_capacity": 120,
                 "is_initial": False},
             {"id": 54, "symbol": "Xe", "name": "Xenon", "base_production_rate": 0.11, "base_storage_capacity": 115,
                 "is_initial": False},
             {"id": 55, "symbol": "Cs", "name": "Caesium", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
             {"id": 56, "symbol": "Ba", "name": "Barium", "base_production_rate": 0.22, "base_storage_capacity": 170,
                 "is_initial": False},
             {"id": 57, "symbol": "La", "name": "Lanthanum", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
             {"id": 58, "symbol": "Ce", "name": "Cerium", "base_production_rate": 0.20, "base_storage_capacity": 160,
                 "is_initial": False},
             {"id": 59, "symbol": "Pr", "name": "Praseodymium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
             {"id": 60, "symbol": "Nd", "name": "Neodymium", "base_production_rate": 0.21,
                 "base_storage_capacity": 165, "is_initial": False},
             {"id": 61, "symbol": "Pm", "name": "Promethium", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
             {"id": 62, "symbol": "Sm", "name": "Samarium", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
             {"id": 63, "symbol": "Eu", "name": "Europium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
             {"id": 64, "symbol": "Gd", "name": "Gadolinium", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
             {"id": 65, "symbol": "Tb", "name": "Terbium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
             {"id": 66, "symbol": "Dy", "name": "Dysprosium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
             {"id": 67, "symbol": "Ho", "name": "Holmium", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
             {"id": 68, "symbol": "Er", "name": "Erbium", "base_production_rate": 0.18, "base_storage_capacity": 150,
                 "is_initial": False},
             {"id": 69, "symbol": "Tm", "name": "Thulium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
             {"id": 70, "symbol": "Yb", "name": "Ytterbium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
             {"id": 71, "symbol": "Lu", "name": "Lutetium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
             {"id": 72, "symbol": "Hf", "name": "Hafnium", "base_production_rate": 0.21,
                 "base_storage_capacity": 165, "is_initial": False},
             {"id": 73, "symbol": "Ta", "name": "Tantalum", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
             {"id": 74, "symbol": "W", "name": "Tungsten", "base_production_rate": 0.25,
                 "base_storage_capacity": 195, "is_initial": False},
             {"id": 75, "symbol": "Re", "name": "Rhenium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
             {"id": 76, "symbol": "Os", "name": "Osmium", "base_production_rate": 0.18, "base_storage_capacity": 150,
                 "is_initial": False},
             {"id": 77, "symbol": "Ir", "name": "Iridium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
             {"id": 78, "symbol": "Pt", "name": "Platinum", "base_production_rate": 0.20,
                 "base_storage_capacity": 160, "is_initial": False},
             {"id": 79, "symbol": "Au", "name": "Gold", "base_production_rate": 0.15, "base_storage_capacity": 135,
                 "is_initial": False},
             {"id": 80, "symbol": "Hg", "name": "Mercury", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
             {"id": 81, "symbol": "Tl", "name": "Thallium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
             {"id": 82, "symbol": "Pb", "name": "Lead", "base_production_rate": 0.22, "base_storage_capacity": 170,
                 "is_initial": False},
             {"id": 83, "symbol": "Bi", "name": "Bismuth", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
             {"id": 84, "symbol": "Po", "name": "Polonium", "base_production_rate": 0.12,
                 "base_storage_capacity": 120, "is_initial": False},
             {"id": 85, "symbol": "At", "name": "Astatine", "base_production_rate": 0.11,
                 "base_storage_capacity": 115, "is_initial": False},
             {"id": 86, "symbol": "Rn", "name": "Radon", "base_production_rate": 0.10, "base_storage_capacity": 110,
                 "is_initial": False},
             {"id": 87, "symbol": "Fr", "name": "Francium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
             {"id": 88, "symbol": "Ra", "name": "Radium", "base_production_rate": 0.17, "base_storage_capacity": 145,
                 "is_initial": False},
             {"id": 89, "symbol": "Ac", "name": "Actinium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
             {"id": 90, "symbol": "Th", "name": "Thorium", "base_production_rate": 0.23,
                 "base_storage_capacity": 180, "is_initial": False},
             {"id": 91, "symbol": "Pa", "name": "Protactinium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
             {"id": 92, "symbol": "U", "name": "Uranium", "base_production_rate": 0.26, "base_storage_capacity": 200,
                 "is_initial": False},
             {"id": 93, "symbol": "Np", "name": "Neptunium", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
             {"id": 94, "symbol": "Pu", "name": "Plutonium", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
             {"id": 95, "symbol": "Am", "name": "Americium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
             {"id": 96, "symbol": "Cm", "name": "Curium", "base_production_rate": 0.17, "base_storage_capacity": 145,
                 "is_initial": False},
             {"id": 97, "symbol": "Bk", "name": "Berkelium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
             {"id": 98, "symbol": "Cf", "name": "Californium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
             {"id": 99, "symbol": "Es", "name": "Einsteinium", "base_production_rate": 0.12,
                 "base_storage_capacity": 120, "is_initial": False},
             {"id": 100, "symbol": "Fm", "name": "Fermium", "base_production_rate": 0.14,
                  "base_storage_capacity": 130, "is_initial": False},
             {"id": 101, "symbol": "Md", "name": "Mendelevium", "base_production_rate": 0.11,
                  "base_storage_capacity": 115, "is_initial": False},
             {"id": 102, "symbol": "No", "name": "Nobelium", "base_production_rate": 0.13,
                  "base_storage_capacity": 125, "is_initial": False},
             {"id": 103, "symbol": "Lr", "name": "Lawrencium", "base_production_rate": 0.12,
                  "base_storage_capacity": 120, "is_initial": False},
             {"id": 104, "symbol": "Rf", "name": "Rutherfordium", "base_production_rate": 0.15,
                  "base_storage_capacity": 135, "is_initial": False},
             {"id": 105, "symbol": "Db", "name": "Dubnium", "base_production_rate": 0.14,
                  "base_storage_capacity": 130, "is_initial": False},
             {"id": 106, "symbol": "Sg", "name": "Seaborgium", "base_production_rate": 0.16,
                  "base_storage_capacity": 140, "is_initial": False},
             {"id": 107, "symbol": "Bh", "name": "Bohrium", "base_production_rate": 0.13,
                  "base_storage_capacity": 125, "is_initial": False},
             {"id": 108, "symbol": "Hs", "name": "Hassium", "base_production_rate": 0.17,
                  "base_storage_capacity": 145, "is_initial": False},
             {"id": 109, "symbol": "Mt", "name": "Meitnerium", "base_production_rate": 0.12,
                  "base_storage_capacity": 120, "is_initial": False},
             {"id": 110, "symbol": "Ds", "name": "Darmstadtium", "base_production_rate": 0.14,
                  "base_storage_capacity": 130, "is_initial": False},
             {"id": 111, "symbol": "Rg", "name": "Roentgenium", "base_production_rate": 0.13,
                  "base_storage_capacity": 125, "is_initial": False},
             {"id": 112, "symbol": "Cn", "name": "Copernicium", "base_production_rate": 0.15,
                  "base_storage_capacity": 135, "is_initial": False},
             {"id": 113, "symbol": "Nh", "name": "Nihonium", "base_production_rate": 0.11,
                  "base_storage_capacity": 115, "is_initial": False},
             {"id": 114, "symbol": "Fl", "name": "Flerovium", "base_production_rate": 0.12,
                  "base_storage_capacity": 120, "is_initial": False},
             {"id": 115, "symbol": "Mc", "name": "Moscovium", "base_production_rate": 0.10,
                  "base_storage_capacity": 110, "is_initial": False},
             {"id": 116, "symbol": "Lv", "name": "Livermorium", "base_production_rate": 0.11,
                  "base_storage_capacity": 115, "is_initial": False},
             {"id": 117, "symbol": "Ts", "name": "Tennessine", "base_production_rate": 0.09,
                  "base_storage_capacity": 105, "is_initial": False},
             {"id": 118, "symbol": "Og", "name": "Oganesson", "base_production_rate": 0.08,
                  "base_storage_capacity": 100, "is_initial": False}
                # ... добавь остальные элементы из server.py
            ]

            for element_data in elements_data:
                element = Element(**element_data)
                db.add(element)

            # Требования для разблокировки элементов
            requirements_data = [
                {"element_id": 3, "required_element_id": 1, "required_amount": 10.0},  # Li требует H
                {"element_id": 3, "required_element_id": 2, "required_amount": 8.0},  # Li требует He
                {"element_id": 4, "required_element_id": 1, "required_amount": 15.0},  # Be требует H
            ]

            for req_data in requirements_data:
                requirement = ElementUnlockRequirement(**req_data)
                db.add(requirement)

            db.commit()
            print(f"✅ Добавлено {len(elements_data)} элементов и {len(requirements_data)} требований")
        else:
            print(f"✅ В базе уже есть {len(existing_elements)} элементов")


def main():
    """Основная функция"""
    print("🚀 Запуск создания таблиц...")
    print(f"📊 База данных: {config.DATABASE_URL}")

    try:
        # Создаем таблицы
        if create_tables():
            # Добавляем начальные данные
            add_initial_data()
            print("\n🎉 Все операции завершены успешно!")
        else:
            print("\n💥 Создание таблиц завершено с ошибками!")

    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")


if __name__ == "__main__":
    main()