#date: 2025-04-23T17:02:24Z
#url: https://api.github.com/gists/a3df57f2318232b894156fd0bb9189a4
#owner: https://api.github.com/users/gazzem-altgr

productos = {}
ventas = []
pedidos = []

def agregar_producto():
    while True:
        nombre = input("ingrese el nombre del producto (o 'fin' para terminar): ")
        if nombre.lower() == 'fin':
            break
        precio = float(input(f"ingrese el precio de {nombre}: "))
        cantidad = int(input(f"ingrese la cantidad en stock de {nombre}: "))
        productos[nombre] = {"precio": precio, "cantidad": cantidad}
    print("productos registrados correctamente.\n")

def mostrar_inventario():
    print("\ninventario actual:")
    for nombre, datos in productos.items():
        print(f"{nombre} - precio: s/{datos['precio']} - cantidad en inventario: {datos['cantidad']}")

def procesar_venta():
    while True:
        cliente = input("ingrese el nombre del cliente (o 'fin' para terminar): ")
        if cliente.lower() == 'fin':
            print("fin de las ventas.")
            break

        productos_venta = []
        print(f"\n   registro de venta para {cliente}    ")

        while True:
            mostrar_inventario()
            nombre_producto = input("ingrese el nombre del producto que desea comprar (o 'fin' para terminar): ")
            if nombre_producto.lower() == 'fin':
                break
            if nombre_producto in productos:
                cantidad = int(input(f"¿cuantas unidades de {nombre_producto} desea?: "))
                if productos[nombre_producto]["cantidad"] >= cantidad:
                    productos_venta.append((nombre_producto, cantidad))
                    productos[nombre_producto]["cantidad"] -= cantidad
                    print(f"{cantidad} unidades de {nombre_producto} agregadas al pedido.")
                else:
                    print(f"no hay suficiente stock de {nombre_producto}.")
            else:
                print(f"el producto {nombre_producto} no esta disponible en inventario.")

        if productos_venta:
            total_venta = sum(productos[nombre]["precio"] * cantidad for nombre, cantidad in productos_venta)
            print(f"\nventa procesada para {cliente}. total de la venta: s/{total_venta}")
            ventas.append({"cliente": cliente, "productos": productos_venta, "total": total_venta})
        else:
            print(f"no se registro ninguna venta para {cliente}.")

def registrar_pedido(cliente):
    productos_pedido_validos = []
    print(f"\n--- registro de pedido para {cliente} ---")
    while True:
        nombre_producto = input("ingrese el nombre del producto (o 'fin' para terminar): ")
        if nombre_producto.lower() == 'fin':
            break
        if nombre_producto in productos:
            cantidad = int(input(f"cuantas unidades de {nombre_producto} desea?: "))
            if productos[nombre_producto]["cantidad"] >= cantidad:
                productos_pedido_validos.append((nombre_producto, cantidad))
                print(f"producto {nombre_producto} agregado al pedido.")
            else:
                print(f"no hay suficiente stock de {nombre_producto}.")
        else:
            print(f"el producto {nombre_producto} no esta disponible en inventario.")
    if productos_pedido_validos:
        pedido = {"cliente": cliente, "productos": productos_pedido_validos, "estado": "pendiente"}
        pedidos.append(pedido)
        mostrar_pedido(pedido)
        for nombre_producto, cantidad in productos_pedido_validos:
            productos[nombre_producto]["cantidad"] -= cantidad
        return True

def notificar_pedido(pedido):
    pedido["estado"] = "listo"
    print(f"el pedido de {pedido['cliente']} esta listo para ser recogido.")

def mostrar_pedido(pedido):
    productos_str = ', '.join([f"{producto[0]} (cantidad: {producto[1]})" for producto in pedido["productos"]])
    print(f"pedido de {pedido['cliente']}: {productos_str}. estado: {pedido['estado']}")

def generar_reporte_ventas():
    total_ventas = sum([venta["total"] for venta in ventas])
    print(f"\nreporte de ventas: total de ventas realizadas: s/{total_ventas}")
    for venta in ventas:
        print(f"cliente: {venta['cliente']} - productos: {', '.join([f'{producto[0]} (cantidad: {producto[1]})' for producto in venta['productos']])} - total: s/{venta['total']}")

def generar_reporte_inventario():
    print("\nreporte de inventario:")
    mostrar_inventario()

agregar_producto()
procesar_venta()
generar_reporte_ventas()
generar_reporte_inventario()