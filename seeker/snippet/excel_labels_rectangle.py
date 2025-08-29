#date: 2025-08-29T16:52:19Z
#url: https://api.github.com/gists/d78126c3b491c665d5de0099b295acb1
#owner: https://api.github.com/users/suonosb

import pandas as pd

def generar_labelstudio_config(excel_path, col_sku="sku", output_file="config.xml"):
    """
    Genera un archivo XML para Label Studio con RectangleLabels
    a partir de una columna de SKUs en un Excel.
    
    Args:
        excel_path (str): Ruta del archivo Excel.
        col_sku (str): Nombre de la columna donde están los SKUs/nombres.
        output_file (str): Nombre del archivo XML de salida.
    """
    # Leer Excel
    df = pd.read_excel(excel_path)

    if col_sku not in df.columns:
        raise ValueError(f"La columna '{col_sku}' no existe en el Excel")

    # Comenzar estructura XML
    xml = []
    xml.append('<View>')
    xml.append('  <Image name="image" value="$image"/>')
    xml.append('')
    xml.append('  <RectangleLabels name="objects" toName="image">')

    # Agregar etiquetas
    for sku in df[col_sku].dropna().astype(str):
        xml.append(f'    <Label value="{sku}"/>')

    xml.append('  </RectangleLabels>')
    xml.append('</View>')

    # Guardar archivo
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(xml))

    print(f"✅ Configuración generada en {output_file}")


# Ejemplo de uso:
# generar_labelstudio_config("skus.xlsx", col_sku="producto", output_file="config.xml")
