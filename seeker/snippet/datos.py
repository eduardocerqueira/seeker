#date: 2026-02-26T17:46:14Z
#url: https://api.github.com/gists/755726de1a781c847ccf2979945bfe1f
#owner: https://api.github.com/users/bormolina

from datetime import datetime, timedelta
from Mascota import Mascota

def get_datos():
    hoy = datetime.today()

    mascotas = [
        Mascota(1, "Misu", "Gato", hoy - timedelta(days=20), 120.0,
                hoy - timedelta(days=400), None, ["Rabia"]),
        Mascota(2, "Simba", "Gato", hoy - timedelta(days=40), 140.0,
                hoy - timedelta(days=500), hoy - timedelta(days=5), ["Rabia"]),
        Mascota(3, "Rocky", "Chihuahua", hoy - timedelta(days=10), 350.0,
                hoy - timedelta(days=800), None, []),
        Mascota(4, "Bella", "Perro", hoy - timedelta(days=30), 300.0,
                hoy - timedelta(days=900), hoy - timedelta(days=10), ["Rabia", "Parvovirus"]),
        Mascota(5, "Pincho", "Erizo", hoy - timedelta(days=5), 80.0,
                hoy - timedelta(days=300), None, ["Antiparasitaria"]),
        Mascota(6, "Thor", "Conejo", hoy - timedelta(days=25), 70.0,
                hoy - timedelta(days=350), hoy - timedelta(days=3), ["Mixomatosis"]),
        Mascota(7, "Venom", "Tarántula", hoy - timedelta(days=3), 45.5,
                hoy - timedelta(days=365), None, None),
        Mascota(8, "Nala", "Gato", hoy - timedelta(days=50), 160.0,
                hoy - timedelta(days=600), hoy - timedelta(days=15), ["Rabia", "Trivalente"]),
        Mascota(9, "Luna", "Gato", hoy - timedelta(days=15), 150.0,
                hoy - timedelta(days=500), None, ["Rabia", "Trivalente"]),
        Mascota(10, "Rex", "Perro", hoy - timedelta(days=35), 250.0,
                hoy - timedelta(days=800), hoy - timedelta(days=7), ["Rabia"]),
        Mascota(11, "Toby", "Perro", hoy - timedelta(days=12), 200.0,
                hoy - timedelta(days=600), None, []),
        Mascota(12, "Kiwi", "Erizo", hoy - timedelta(days=45), 85.0,
                hoy - timedelta(days=330), hoy - timedelta(days=12), ["Antiparasitaria"]),
        Mascota(13, "Nina", "Conejo", hoy - timedelta(days=8), 60.0,
                hoy - timedelta(days=200), None, ["Mixomatosis"]),
        Mascota(14, "Loki", "Tarántula", hoy - timedelta(days=20), 50.0,
                hoy - timedelta(days=400), hoy - timedelta(days=1), None),
        Mascota(15, "Spike", "Erizo", hoy - timedelta(days=18), 75.0,
                hoy - timedelta(days=320), None, ["Antiparasitaria"]),
        Mascota(16, "Daisy", "Conejo", hoy - timedelta(days=28), 65.0,
                hoy - timedelta(days=370), hoy - timedelta(days=9), ["Mixomatosis"]),
        Mascota(17, "Kira", "Gato", hoy - timedelta(days=6), 130.0,
                hoy - timedelta(days=450), None, ["Rabia"]),
        Mascota(18, "Milo", "Gato", hoy - timedelta(days=60), 155.0,
                hoy - timedelta(days=700), hoy - timedelta(days=20), ["Rabia"]),
        Mascota(19, "Max", "Perro", hoy - timedelta(days=2), 220.0,
                hoy - timedelta(days=700), None, ["Rabia", "Parvovirus"]),
        Mascota(20, "Bruno", "Perro", hoy - timedelta(days=55), 280.0,
                hoy - timedelta(days=850), hoy - timedelta(days=18), ["Rabia", "Parvovirus"]),
    ]

    return mascotas


from datetime import datetime, timedelta
from Producto import Producto


def get_productos():
    hoy = datetime(2026, 2, 23)

    productos = [
        Producto(1, "Leche entera", ["Lácteos", "Refrigerados"], 1.25,
                 hoy - timedelta(days=2), hoy + timedelta(days=5)),

        Producto(2, "Yogur natural", ["Lácteos", "Refrigerados"], 0.95,
                 hoy - timedelta(days=7), hoy + timedelta(days=2)),

        Producto(3, "Pan de molde", ["Panadería", "Despensa"], 1.60,
                 hoy - timedelta(days=1), hoy + timedelta(days=6)),

        Producto(4, "Jamón cocido", ["Charcutería", "Refrigerados"], 2.80,
                 hoy - timedelta(days=5), hoy + timedelta(days=10)),

        Producto(5, "Pechuga de pavo", ["Charcutería", "Proteína"], 2.40,
                 hoy - timedelta(days=3), hoy + timedelta(days=8)),

        Producto(6, "Atún en lata", ["Conservas", "Proteína"], 1.10,
                 hoy - timedelta(days=30), hoy + timedelta(days=365)),

        Producto(7, "Tomate frito", ["Conservas", "Salsas"], 1.35,
                 hoy - timedelta(days=15), hoy + timedelta(days=120)),

        Producto(8, "Manzana", ["Fruta", "Fresco"], 0.55,
                 hoy - timedelta(days=2), hoy + timedelta(days=12)),

        Producto(9, "Plátano", ["Fruta", "Fresco"], 0.45,
                 hoy - timedelta(days=3), hoy + timedelta(days=5)),

        Producto(10, "Lechuga", ["Verdura", "Fresco"], 1.05,
                 hoy - timedelta(days=1), hoy + timedelta(days=4)),

        Producto(11, "Pasta", ["Despensa", "Cereales"], 1.20,
                 hoy - timedelta(days=60), hoy + timedelta(days=540)),

        Producto(12, "Arroz", ["Despensa", "Cereales"], 1.10,
                 hoy - timedelta(days=90), hoy + timedelta(days=700)),

        Producto(13, "Chocolate", ["Dulces", "Snack"], 1.75,
                 hoy - timedelta(days=20), hoy + timedelta(days=180)),

        Producto(14, "Zumo de naranja", ["Bebidas", "Refrigerados"], 1.90,
                 hoy - timedelta(days=10), hoy + timedelta(days=25)),

        Producto(15, "Ensalada preparada", ["Refrigerados"], 3.50,
                 hoy - timedelta(days=1), hoy - timedelta(days=3)),
    ]

    return productos