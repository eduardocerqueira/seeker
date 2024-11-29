#date: 2024-11-29T17:06:46Z
#url: https://api.github.com/gists/36cdb4c734c289259424e402264f3d1d
#owner: https://api.github.com/users/codechappie

from PIL import Image, ImageDraw, ImageFont
import os
import ctypes
import time
from datetime import datetime

imagen_original = 'c:/Users/david/Desktop/descarga.jpg'  # Ruta de la imagen original
imagen_modificada = 'c:/Users/david/Desktop/descarga_new.jpg'  # Ruta para guardar la imagen modificada

# Usar una fuente (puedes cambiarla si deseas)
font = ImageFont.truetype("c:/Users/david/Desktop/FiraCodeRegular.ttf", 40)

# Función para agregar texto a una imagen
def agregar_texto_a_imagen():
    now = datetime.now()
    # Texto que deseas agregar
    texto = f"""{{
    "date": {{
        "day": {now.day},
        "month": {now.month},
        "year": {now.year}
    }},
    "time": {{
        "hour": {now.hour},
        "minute": {now.minute},
        "second": {now.second}
   }}
}}"""

    # Reabrir la imagen original para asegurarse de que está limpia
    img = Image.open(imagen_original)  # Abrir la imagen original

    draw = ImageDraw.Draw(img)  # Inicializar el objeto de dibujo

    # Cálculos para posicionar texto al centro
    ancho_imagen, altura_imagen = img.size
    bbox = draw.textbbox((0, 0), texto, font=font)
    ancho_texto = bbox[2] - bbox[0]  # Ancho del bounding box (derecha - izquierda)
    altura_texto = bbox[3] - bbox[1]  # Altura del bounding box (abajo - arriba)
    posicion_x = (ancho_imagen - ancho_texto) // 2  # Centrado horizontal
    posicion_y = (altura_imagen - altura_texto) // 2  # Centrado vertical
    posicion = (posicion_x, posicion_y)  # Posición en la que dibujar el texto
    color_texto = (255, 255, 255)  # Definir el color (Blanco)
    draw.text(posicion, texto, fill=color_texto, font=font)  # Escribir el texto en la imagen

    # Verificar si el archivo ya existe y eliminarlo antes de guardar
    if os.path.exists(imagen_modificada):
        os.remove(imagen_modificada)  # Eliminar el archivo existente
        print("Archivo anterior eliminado.")

    # Guardar la nueva imagen modificada
    img.save(imagen_modificada)

# Función para cambiar el wallpaper (en Windows)
def establecer_como_wallpaper(imagen):
    # Para Windows
    ctypes.windll.user32.SystemParametersInfoW(20, 0, imagen, 3)

# Bucle para cambiar el wallpaper cada segundo
while True:
    start_time = time.time()  # Captura el tiempo al inicio de la iteración

    # Llamar a la función para agregar texto
    agregar_texto_a_imagen()

    # Establecer la imagen modificada como fondo de pantalla
    establecer_como_wallpaper(imagen_modificada)

    # Medir el tiempo transcurrido
    elapsed_time = time.time() - start_time  # Tiempo transcurrido desde el inicio

    # Esperamos el tiempo restante para completar 1 segundo
    if elapsed_time < 1:
        time.sleep(1 - elapsed_time)  # Espera para completar el segundo
    else:
        time.sleep(1)  # Si el tiempo excede 1 segundo, esperamos 1 segundo completo
