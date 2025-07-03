#date: 2025-07-03T16:48:49Z
#url: https://api.github.com/gists/7c991585aca1b274b7bdbd33cb350e4d
#owner: https://api.github.com/users/lauvigne

import pymupdf

def detect_columns(page, min_height=50):
    """
    Retourne une liste de colonnes détectées à partir des lignes verticales (traits graphiques).
    Chaque colonne est définie par (x0, x1).
    """
    drawings = page.get_drawings()
    vertical_lines = []

    for drawing in drawings:
        for item in drawing["items"]:
            if item[0] == "l":  # ligne
                x0, y0, x1, y1 = item[1]
                if abs(x0 - x1) < 2 and abs(y1 - y0) > min_height:
                    # Ligne quasi verticale et assez haute
                    vertical_lines.append((x0, y0, x1, y1))

    # Trier les x pour créer des colonnes
    x_positions = sorted(set([line[0] for line in vertical_lines]))
    columns = []
    for i in range(len(x_positions) - 1):
        x0 = x_positions[i]
        x1 = x_positions[i + 1]
        columns.append((x0, x1))

    return columns

def find_column_for_block(block, columns, margin=2):
    x0_block, _, x1_block, _ = block[:4]

    for idx, (x0_col, x1_col) in enumerate(columns):
        if (x0_block + margin) >= x0_col and (x1_block - margin) <= x1_col:
            return idx
    return None

def assign_blocks_to_columns(page, columns):
    blocks = page.get_text("blocks")
    column_texts = {idx: [] for idx in range(len(columns))}

    for block in blocks:
        col_idx = find_column_for_block(block, columns)
        if col_idx is not None:
            text = block[4].strip()
            if text:
                column_texts[col_idx].append(text)

    return column_texts

# ---------- Exemple d'utilisation ----------

doc = pymupdf.open("xxxxx.pdf")
page = doc[0]

columns = detect_columns(page)
print("Colonnes détectées :", columns)

column_texts = assign_blocks_to_columns(page, columns)

for col_idx, texts in column_texts.items():
    print(f"\n Colonne {col_idx + 1} :")
    for txt in texts:
        print(f"  - {txt}")