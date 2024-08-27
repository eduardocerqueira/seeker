#date: 2024-08-27T16:59:28Z
#url: https://api.github.com/gists/7f7b7249869b6a11fbba48c57cff19e1
#owner: https://api.github.com/users/aspose-com-kb

import aspose.words as aw
import aspose.pydrawing as drawing

def mergeCells(startCell: aw.tables.Cell, endCell: aw.tables.Cell):

    parentTable = startCell.parent_row.parent_table

    # Find the start and end cell position
    startCellPos = drawing.Point(startCell.parent_row.index_of(startCell), parentTable.index_of(startCell.parent_row))
    endCellPos = drawing.Point(endCell.parent_row.index_of(endCell), parentTable.index_of(endCell.parent_row))

    # Create a range of cells
    mergeRange = drawing.Rectangle(
        min(startCellPos.x, endCellPos.x),
        min(startCellPos.y, endCellPos.y),
        abs(endCellPos.x - startCellPos.x) + 1,
        abs(endCellPos.y - startCellPos.y) + 1)

    for row in parentTable.rows:
        row = row.as_row()
        for cell in row.cells:
            cell = cell.as_cell()
            currentPos = drawing.Point(row.index_of(cell), parentTable.index_of(row))

            # Merge the cell if inside the range
            if mergeRange.contains(currentPos):
                cell.cell_format.horizontal_merge = aw.tables.CellMerge.FIRST if currentPos.x == mergeRange.x else aw.tables.CellMerge.PREVIOUS
                cell.cell_format.vertical_merge = aw.tables.CellMerge.FIRST if currentPos.y == mergeRange.y else aw.tables.CellMerge.PREVIOUS

# Load the license
wordLic = aw.License()
wordLic.set_license("license.lic")

tableDoc = aw.Document("Table.docx")

table = tableDoc.first_section.body.tables[0]

# Define start and end cell for the range
cellStartRange = table.rows[0].cells[0]
cellEndRange = table.rows[1].cells[1]

# Merge cells
mergeCells(cellStartRange, cellEndRange)

tableDoc.save("output.docx")
    
print ("Table cells merged successfully")