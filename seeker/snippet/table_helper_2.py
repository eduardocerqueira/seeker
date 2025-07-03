#date: 2025-07-03T16:48:49Z
#url: https://api.github.com/gists/7c991585aca1b274b7bdbd33cb350e4d
#owner: https://api.github.com/users/lauvigne

import pymupdf

def detect_columns(page, min_height=50):
    """
    Retourne une liste de tuples (x0, x1) représentant les colonnes,
    détectées à partir des lignes verticales graphiques.
    """
    drawings = page.get_drawings()
    vertical_lines = []

    for drawing in drawings:
        for item in drawing["items"]:
            if item[0] == "l":
                x0, y0, x1, y1 = item[1]
                if abs(x0 - x1) < 2 and abs(y1 - y0) > min_height:
                    vertical_lines.append((x0, y0, x1, y1))

    # Trier les positions X
    x_positions = sorted(set([round(line[0]) for line in vertical_lines]))
    columns = []
    for i in range(len(x_positions) - 1):
        columns.append((x_positions[i], x_positions[i + 1]))
    return columns

def create_cell_bboxes(columns, table_y0, table_y1, num_rows):
    """
    Retourne une liste de bbox pour toutes les cellules :
    (x0, y0, x1, y1) pour chaque cellule.
    """
    row_height = (table_y1 - table_y0) / num_rows
    cells = []

    for i in range(num_rows):
        y0_row = table_y0 + i * row_height
        y1_row = y0_row + row_height

        for x0_col, x1_col in columns:
            cell_bbox = (x0_col, y0_row, x1_col, y1_row)
            cells.append(cell_bbox)

    return cells
  
def find_cell_for_block(block_bbox, cell_bboxes, tolerance=2):
    """
    Trouve la bbox de la cellule qui contient le bloc texte donné.
    Retourne la bbox de la cellule ou None si aucune cellule ne contient le bloc.
    """
    x0_block, y0_block, x1_block, y1_block = block_bbox

    for cell_bbox in cell_bboxes:
        x0_cell, y0_cell, x1_cell, y1_cell = cell_bbox

        if (x0_block >= x0_cell - tolerance and
            x1_block <= x1_cell + tolerance and
            y0_block >= y0_cell - tolerance and
            y1_block <= y1_cell + tolerance):
            return cell_bbox

    return None  
    