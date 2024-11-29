#date: 2024-11-29T17:00:09Z
#url: https://api.github.com/gists/f2dcd4bb45447602fdd8f508199f46ca
#owner: https://api.github.com/users/codechappie

from PIL import Image, ImageDraw, ImageFont
import ctypes, os, time, io
from datetime import datetime

background_color = (0, 0, 0)
text_color = (22, 230, 69)
font_size = 30
font_path = r"c:/Users/david/Desktop/FiraCodeRegular.ttf"
font = ImageFont.truetype(font_path, font_size)

screen_width = 1920
screen_height = 1080

def two_digits_format(number):
    return f"0{number}" if number < 10 else number

# Función para agregar texto a una imagen con fondo sólido
def agregar_texto_a_imagen():
    now = datetime.now()

    texto = f"""{{
    "date": {{
        "day": {two_digits_format(now.day)},
        "month": {two_digits_format(now.month)},
        "year": {now.year}
    }},
    "time": {{
        "hour": {two_digits_format(now.hour)},
        "minute": {two_digits_format(now.minute)}
        "second":  {two_digits_format(now.second)},
        "meridiem": "{now.strftime("%p")}"
    }}
}}"""

    # Crear una imagen con fondo sólido
    img = Image.new("RGB", (screen_width, screen_height), background_color)
    draw = ImageDraw.Draw(img)  # Inicializar el objeto de dibujo

    # Cálculos para posicionar texto al centro
    ancho_imagen, altura_imagen = img.size
    bbox = draw.textbbox((0, 0), texto, font=font)
    ancho_texto = bbox[2] - bbox[0]
    altura_texto = bbox[3] - bbox[1]
    posicion_x = (ancho_imagen - ancho_texto) // 2  # Centrado horizontal
    posicion_y = (altura_imagen - altura_texto) // 2  # Centrado vertical
    posicion = (posicion_x, posicion_y)

    # Escribir el texto en la imagen con fondo sólido
    draw.text(posicion, texto, fill=text_color, font=font)

    # Crear un archivo temporal en memoria y devolverlo
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="BMP")
    img_bytes.seek(0)

    return img_bytes


# Función para cambiar el wallpaper (en Windows)
def establecer_como_wallpaper(imagen_bytes):
    # Guardamos la imagen temporalmente
    temp_path = r"C:/Users/david/Pictures/temp_wallpaper.bmp"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    with open(temp_path, "wb") as f:
        f.write(imagen_bytes.read())

    ctypes.windll.user32.SystemParametersInfoW(
        20, 0, temp_path, 3
    )  # Establecer la imagen temporal


while True:
    start_time = time.time()
    imagen_bytes = agregar_texto_a_imagen()
    establecer_como_wallpaper(imagen_bytes)
    elapsed_time = time.time() - start_time

    if elapsed_time < 1:
        time.sleep(1 - elapsed_time)  # Espera para completar el segundo
    else:
        time.sleep(1)  # Si el tiempo excede 1 segundo, esperamos 1 segundo completo
