#date: 2025-10-28T17:10:24Z
#url: https://api.github.com/gists/a774170a69cb481f24168bb3439a3022
#owner: https://api.github.com/users/mix0073

from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn

from database import engine, AsyncSessionLocal  # Исправлен импорт
from rotate import start_background_processor
from endpoints import router as api_router


# Тестовые данные (временные)
elements_db = {}
recipes_db = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_test_data()
    # Запускаем фоновый процессор (теперь с БД)
    start_background_processor()
    print("✅ Сервер запущен на http://localhost:8000")
    print("✅ Работает с сохранением в БД")
    yield
    # Shutdown
    print("🛑 Сервер остановлен")


app = FastAPI(lifespan=lifespan)

# Подключаем роутеры
app.include_router(api_router)


def init_test_data():
    global elements_db, recipes_db

    elements_db = {

            1: {"id": 1, "symbol": "H", "name": "Hydrogen", "base_production_rate": 0.15, "base_storage_capacity": 120,
                "is_initial": True},
            2: {"id": 2, "symbol": "He", "name": "Helium", "base_production_rate": 0.12, "base_storage_capacity": 100,
                "is_initial": True},
            3: {"id": 3, "symbol": "Li", "name": "Lithium", "base_production_rate": 0.18, "base_storage_capacity": 150,
                "is_initial": False},
            4: {"id": 4, "symbol": "Be", "name": "Beryllium", "base_production_rate": 0.16,
                "base_storage_capacity": 140, "is_initial": False},
            5: {"id": 5, "symbol": "B", "name": "Boron", "base_production_rate": 0.14, "base_storage_capacity": 130,
                "is_initial": False},
            6: {"id": 6, "symbol": "C", "name": "Carbon", "base_production_rate": 0.20, "base_storage_capacity": 160,
                "is_initial": True},
            7: {"id": 7, "symbol": "N", "name": "Nitrogen", "base_production_rate": 0.17, "base_storage_capacity": 145,
                "is_initial": True},
            8: {"id": 8, "symbol": "O", "name": "Oxygen", "base_production_rate": 0.22, "base_storage_capacity": 180,
                "is_initial": True},
            9: {"id": 9, "symbol": "F", "name": "Fluorine", "base_production_rate": 0.13, "base_storage_capacity": 125,
                "is_initial": False},
            10: {"id": 10, "symbol": "Ne", "name": "Neon", "base_production_rate": 0.11, "base_storage_capacity": 110,
                 "is_initial": False},
            11: {"id": 11, "symbol": "Na", "name": "Sodium", "base_production_rate": 0.19, "base_storage_capacity": 155,
                 "is_initial": False},
            12: {"id": 12, "symbol": "Mg", "name": "Magnesium", "base_production_rate": 0.21,
                 "base_storage_capacity": 165, "is_initial": False},
            13: {"id": 13, "symbol": "Al", "name": "Aluminium", "base_production_rate": 0.23,
                 "base_storage_capacity": 175, "is_initial": False},
            14: {"id": 14, "symbol": "Si", "name": "Silicon", "base_production_rate": 0.25,
                 "base_storage_capacity": 190, "is_initial": False},
            15: {"id": 15, "symbol": "P", "name": "Phosphorus", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
            16: {"id": 16, "symbol": "S", "name": "Sulfur", "base_production_rate": 0.16, "base_storage_capacity": 140,
                 "is_initial": False},
            17: {"id": 17, "symbol": "Cl", "name": "Chlorine", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
            18: {"id": 18, "symbol": "Ar", "name": "Argon", "base_production_rate": 0.12, "base_storage_capacity": 120,
                 "is_initial": False},
            19: {"id": 19, "symbol": "K", "name": "Potassium", "base_production_rate": 0.20,
                 "base_storage_capacity": 160, "is_initial": False},
            20: {"id": 20, "symbol": "Ca", "name": "Calcium", "base_production_rate": 0.22,
                 "base_storage_capacity": 170, "is_initial": False},
            21: {"id": 21, "symbol": "Sc", "name": "Scandium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
            22: {"id": 22, "symbol": "Ti", "name": "Titanium", "base_production_rate": 0.24,
                 "base_storage_capacity": 185, "is_initial": False},
            23: {"id": 23, "symbol": "V", "name": "Vanadium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
            24: {"id": 24, "symbol": "Cr", "name": "Chromium", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
            25: {"id": 25, "symbol": "Mn", "name": "Manganese", "base_production_rate": 0.21,
                 "base_storage_capacity": 165, "is_initial": False},
            26: {"id": 26, "symbol": "Fe", "name": "Iron", "base_production_rate": 0.28, "base_storage_capacity": 220,
                 "is_initial": False},
            27: {"id": 27, "symbol": "Co", "name": "Cobalt", "base_production_rate": 0.16, "base_storage_capacity": 140,
                 "is_initial": False},
            28: {"id": 28, "symbol": "Ni", "name": "Nickel", "base_production_rate": 0.23, "base_storage_capacity": 180,
                 "is_initial": False},
            29: {"id": 29, "symbol": "Cu", "name": "Copper", "base_production_rate": 0.20, "base_storage_capacity": 160,
                 "is_initial": False},
            30: {"id": 30, "symbol": "Zn", "name": "Zinc", "base_production_rate": 0.18, "base_storage_capacity": 150,
                 "is_initial": False},
            31: {"id": 31, "symbol": "Ga", "name": "Gallium", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
            32: {"id": 32, "symbol": "Ge", "name": "Germanium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
            33: {"id": 33, "symbol": "As", "name": "Arsenic", "base_production_rate": 0.12,
                 "base_storage_capacity": 120, "is_initial": False},
            34: {"id": 34, "symbol": "Se", "name": "Selenium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
            35: {"id": 35, "symbol": "Br", "name": "Bromine", "base_production_rate": 0.11,
                 "base_storage_capacity": 115, "is_initial": False},
            36: {"id": 36, "symbol": "Kr", "name": "Krypton", "base_production_rate": 0.10,
                 "base_storage_capacity": 110, "is_initial": False},
            37: {"id": 37, "symbol": "Rb", "name": "Rubidium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
            38: {"id": 38, "symbol": "Sr", "name": "Strontium", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
            39: {"id": 39, "symbol": "Y", "name": "Yttrium", "base_production_rate": 0.16, "base_storage_capacity": 140,
                 "is_initial": False},
            40: {"id": 40, "symbol": "Zr", "name": "Zirconium", "base_production_rate": 0.22,
                 "base_storage_capacity": 170, "is_initial": False},
            41: {"id": 41, "symbol": "Nb", "name": "Niobium", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
            42: {"id": 42, "symbol": "Mo", "name": "Molybdenum", "base_production_rate": 0.24,
                 "base_storage_capacity": 185, "is_initial": False},
            43: {"id": 43, "symbol": "Tc", "name": "Technetium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
            44: {"id": 44, "symbol": "Ru", "name": "Ruthenium", "base_production_rate": 0.20,
                 "base_storage_capacity": 160, "is_initial": False},
            45: {"id": 45, "symbol": "Rh", "name": "Rhodium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
            46: {"id": 46, "symbol": "Pd", "name": "Palladium", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
            47: {"id": 47, "symbol": "Ag", "name": "Silver", "base_production_rate": 0.17, "base_storage_capacity": 145,
                 "is_initial": False},
            48: {"id": 48, "symbol": "Cd", "name": "Cadmium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
            49: {"id": 49, "symbol": "In", "name": "Indium", "base_production_rate": 0.14, "base_storage_capacity": 130,
                 "is_initial": False},
            50: {"id": 50, "symbol": "Sn", "name": "Tin", "base_production_rate": 0.21, "base_storage_capacity": 165,
                 "is_initial": False},
            51: {"id": 51, "symbol": "Sb", "name": "Antimony", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
            52: {"id": 52, "symbol": "Te", "name": "Tellurium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
            53: {"id": 53, "symbol": "I", "name": "Iodine", "base_production_rate": 0.12, "base_storage_capacity": 120,
                 "is_initial": False},
            54: {"id": 54, "symbol": "Xe", "name": "Xenon", "base_production_rate": 0.11, "base_storage_capacity": 115,
                 "is_initial": False},
            55: {"id": 55, "symbol": "Cs", "name": "Caesium", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
            56: {"id": 56, "symbol": "Ba", "name": "Barium", "base_production_rate": 0.22, "base_storage_capacity": 170,
                 "is_initial": False},
            57: {"id": 57, "symbol": "La", "name": "Lanthanum", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
            58: {"id": 58, "symbol": "Ce", "name": "Cerium", "base_production_rate": 0.20, "base_storage_capacity": 160,
                 "is_initial": False},
            59: {"id": 59, "symbol": "Pr", "name": "Praseodymium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
            60: {"id": 60, "symbol": "Nd", "name": "Neodymium", "base_production_rate": 0.21,
                 "base_storage_capacity": 165, "is_initial": False},
            61: {"id": 61, "symbol": "Pm", "name": "Promethium", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
            62: {"id": 62, "symbol": "Sm", "name": "Samarium", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
            63: {"id": 63, "symbol": "Eu", "name": "Europium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
            64: {"id": 64, "symbol": "Gd", "name": "Gadolinium", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
            65: {"id": 65, "symbol": "Tb", "name": "Terbium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
            66: {"id": 66, "symbol": "Dy", "name": "Dysprosium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
            67: {"id": 67, "symbol": "Ho", "name": "Holmium", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
            68: {"id": 68, "symbol": "Er", "name": "Erbium", "base_production_rate": 0.18, "base_storage_capacity": 150,
                 "is_initial": False},
            69: {"id": 69, "symbol": "Tm", "name": "Thulium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
            70: {"id": 70, "symbol": "Yb", "name": "Ytterbium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
            71: {"id": 71, "symbol": "Lu", "name": "Lutetium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
            72: {"id": 72, "symbol": "Hf", "name": "Hafnium", "base_production_rate": 0.21,
                 "base_storage_capacity": 165, "is_initial": False},
            73: {"id": 73, "symbol": "Ta", "name": "Tantalum", "base_production_rate": 0.19,
                 "base_storage_capacity": 155, "is_initial": False},
            74: {"id": 74, "symbol": "W", "name": "Tungsten", "base_production_rate": 0.25,
                 "base_storage_capacity": 195, "is_initial": False},
            75: {"id": 75, "symbol": "Re", "name": "Rhenium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
            76: {"id": 76, "symbol": "Os", "name": "Osmium", "base_production_rate": 0.18, "base_storage_capacity": 150,
                 "is_initial": False},
            77: {"id": 77, "symbol": "Ir", "name": "Iridium", "base_production_rate": 0.17,
                 "base_storage_capacity": 145, "is_initial": False},
            78: {"id": 78, "symbol": "Pt", "name": "Platinum", "base_production_rate": 0.20,
                 "base_storage_capacity": 160, "is_initial": False},
            79: {"id": 79, "symbol": "Au", "name": "Gold", "base_production_rate": 0.15, "base_storage_capacity": 135,
                 "is_initial": False},
            80: {"id": 80, "symbol": "Hg", "name": "Mercury", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
            81: {"id": 81, "symbol": "Tl", "name": "Thallium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
            82: {"id": 82, "symbol": "Pb", "name": "Lead", "base_production_rate": 0.22, "base_storage_capacity": 170,
                 "is_initial": False},
            83: {"id": 83, "symbol": "Bi", "name": "Bismuth", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
            84: {"id": 84, "symbol": "Po", "name": "Polonium", "base_production_rate": 0.12,
                 "base_storage_capacity": 120, "is_initial": False},
            85: {"id": 85, "symbol": "At", "name": "Astatine", "base_production_rate": 0.11,
                 "base_storage_capacity": 115, "is_initial": False},
            86: {"id": 86, "symbol": "Rn", "name": "Radon", "base_production_rate": 0.10, "base_storage_capacity": 110,
                 "is_initial": False},
            87: {"id": 87, "symbol": "Fr", "name": "Francium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
            88: {"id": 88, "symbol": "Ra", "name": "Radium", "base_production_rate": 0.17, "base_storage_capacity": 145,
                 "is_initial": False},
            89: {"id": 89, "symbol": "Ac", "name": "Actinium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
            90: {"id": 90, "symbol": "Th", "name": "Thorium", "base_production_rate": 0.23,
                 "base_storage_capacity": 180, "is_initial": False},
            91: {"id": 91, "symbol": "Pa", "name": "Protactinium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
            92: {"id": 92, "symbol": "U", "name": "Uranium", "base_production_rate": 0.26, "base_storage_capacity": 200,
                 "is_initial": False},
            93: {"id": 93, "symbol": "Np", "name": "Neptunium", "base_production_rate": 0.14,
                 "base_storage_capacity": 130, "is_initial": False},
            94: {"id": 94, "symbol": "Pu", "name": "Plutonium", "base_production_rate": 0.18,
                 "base_storage_capacity": 150, "is_initial": False},
            95: {"id": 95, "symbol": "Am", "name": "Americium", "base_production_rate": 0.15,
                 "base_storage_capacity": 135, "is_initial": False},
            96: {"id": 96, "symbol": "Cm", "name": "Curium", "base_production_rate": 0.17, "base_storage_capacity": 145,
                 "is_initial": False},
            97: {"id": 97, "symbol": "Bk", "name": "Berkelium", "base_production_rate": 0.13,
                 "base_storage_capacity": 125, "is_initial": False},
            98: {"id": 98, "symbol": "Cf", "name": "Californium", "base_production_rate": 0.16,
                 "base_storage_capacity": 140, "is_initial": False},
            99: {"id": 99, "symbol": "Es", "name": "Einsteinium", "base_production_rate": 0.12,
                 "base_storage_capacity": 120, "is_initial": False},
            100: {"id": 100, "symbol": "Fm", "name": "Fermium", "base_production_rate": 0.14,
                  "base_storage_capacity": 130, "is_initial": False},
            101: {"id": 101, "symbol": "Md", "name": "Mendelevium", "base_production_rate": 0.11,
                  "base_storage_capacity": 115, "is_initial": False},
            102: {"id": 102, "symbol": "No", "name": "Nobelium", "base_production_rate": 0.13,
                  "base_storage_capacity": 125, "is_initial": False},
            103: {"id": 103, "symbol": "Lr", "name": "Lawrencium", "base_production_rate": 0.12,
                  "base_storage_capacity": 120, "is_initial": False},
            104: {"id": 104, "symbol": "Rf", "name": "Rutherfordium", "base_production_rate": 0.15,
                  "base_storage_capacity": 135, "is_initial": False},
            105: {"id": 105, "symbol": "Db", "name": "Dubnium", "base_production_rate": 0.14,
                  "base_storage_capacity": 130, "is_initial": False},
            106: {"id": 106, "symbol": "Sg", "name": "Seaborgium", "base_production_rate": 0.16,
                  "base_storage_capacity": 140, "is_initial": False},
            107: {"id": 107, "symbol": "Bh", "name": "Bohrium", "base_production_rate": 0.13,
                  "base_storage_capacity": 125, "is_initial": False},
            108: {"id": 108, "symbol": "Hs", "name": "Hassium", "base_production_rate": 0.17,
                  "base_storage_capacity": 145, "is_initial": False},
            109: {"id": 109, "symbol": "Mt", "name": "Meitnerium", "base_production_rate": 0.12,
                  "base_storage_capacity": 120, "is_initial": False},
            110: {"id": 110, "symbol": "Ds", "name": "Darmstadtium", "base_production_rate": 0.14,
                  "base_storage_capacity": 130, "is_initial": False},
            111: {"id": 111, "symbol": "Rg", "name": "Roentgenium", "base_production_rate": 0.13,
                  "base_storage_capacity": 125, "is_initial": False},
            112: {"id": 112, "symbol": "Cn", "name": "Copernicium", "base_production_rate": 0.15,
                  "base_storage_capacity": 135, "is_initial": False},
            113: {"id": 113, "symbol": "Nh", "name": "Nihonium", "base_production_rate": 0.11,
                  "base_storage_capacity": 115, "is_initial": False},
            114: {"id": 114, "symbol": "Fl", "name": "Flerovium", "base_production_rate": 0.12,
                  "base_storage_capacity": 120, "is_initial": False},
            115: {"id": 115, "symbol": "Mc", "name": "Moscovium", "base_production_rate": 0.10,
                  "base_storage_capacity": 110, "is_initial": False},
            116: {"id": 116, "symbol": "Lv", "name": "Livermorium", "base_production_rate": 0.11,
                  "base_storage_capacity": 115, "is_initial": False},
            117: {"id": 117, "symbol": "Ts", "name": "Tennessine", "base_production_rate": 0.09,
                  "base_storage_capacity": 105, "is_initial": False},
            118: {"id": 118, "symbol": "Og", "name": "Oganesson", "base_production_rate": 0.08,
                  "base_storage_capacity": 100, "is_initial": False}

    }

    recipes_db = {
        3: [
            {"required_element_id": 1, "required_amount": 10.0},
            {"required_element_id": 2, "required_amount": 8.0}
        ]
    }

    print("✅ Тестовые данные инициализированы")


@app.get("/")
def root():
    return {
        "message": "Cube Game API",
        "status": "running",
        "mode": "modular structure"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")