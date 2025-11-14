#date: 2025-11-14T16:54:53Z
#url: https://api.github.com/gists/85048487b26b5369c9a0a61b62a23792
#owner: https://api.github.com/users/tello2004

from flask import Flask, jsonify, request

application = Flask(__name__)

db_products = {
    "P01": {"nombre": "Laptop Pro", "precio": 1500.00},
    "P02": {"nombre": "Mouse Inalámbrico", "precio": 45.50},
    "P03": {"nombre": "Teclado Mecánico", "precio": 120.00}
}

db_cart = {} 
db_orders = []


@application.route('/')
def index():
    return jsonify({"mensaje": "Bienvenido a la API de la Tienda en Línea"})


@application.route('/catalogo', methods=['GET'])
def get_catalogo():
    return jsonify({
        "status": "success",
        "productos": db_products
    })


@application.route('/carrito/agregar', methods=['POST'])
def add_to_cart():
    
    data = request.json
    item_id = data.get('item_id')
    cantidad = int(data.get('cantidad', 1))

    if not item_id or item_id not in db_products:
        return jsonify({"error": "ID de producto inválido"}), 404

    
    if item_id in db_cart:
        db_cart[item_id] += cantidad
    else:
        db_cart[item_id] = cantidad

    return jsonify({
        "mensaje": f"Producto {item_id} agregado al carrito.",
        "carrito_actual": db_cart
    }), 201


@application.route('/checkout', methods=['POST'])
def checkout():
    if not db_cart:
        return jsonify({"error": "El carrito está vacío"}), 400

    total_orden = 0
    detalles_orden = []

    
    for item_id, cantidad in db_cart.items():
        producto = db_products[item_id]
        subtotal = producto["precio"] * cantidad
        total_orden += subtotal
        detalles_orden.append({
            "producto": producto["nombre"],
            "item_id": item_id,
            "cantidad": cantidad,
            "subtotal": subtotal
        })

    
    nueva_orden = {
        "orden_id": f"ORD-{len(db_orders) + 1}",
        "detalles": detalles_orden,
        "total_pagado": total_orden
    }

    db_orders.append(nueva_orden)
    db_cart.clear() 

    return jsonify({
        "mensaje": "Compra realizada con éxito",
        "orden": nueva_orden
    }), 201

if __name__ == '__main__':
    application.run(debug=True)