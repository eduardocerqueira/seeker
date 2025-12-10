#date: 2025-12-10T16:54:36Z
#url: https://api.github.com/gists/80fe6e1f68cf245e8a06620e5167f7e7
#owner: https://api.github.com/users/aspose-com-kb

import jpype
import asposecells
jpype.startJVM()

from asposecells.api import (
    License, Workbook, SaveFormat,CellBorderType,
    ChartType, FormattingType, Color, LegendPositionType
)

# 1) Workbook & data
lic = License()
lic.setLicense("license.lic")

wb = Workbook()
ws = wb.getWorksheets().get(0)
ws.setName("Data")

cells = ws.getCells()
# Headers (Level 1..3 + Value)
cells.get("A1").putValue("Level1")
cells.get("B1").putValue("Level2")
cells.get("C1").putValue("Level3")
cells.get("D1").putValue("Value")

# Sample hierarchy (Region -> Country -> City -> Value)
rows = [
    ("Americas","USA","New York",120),
    ("Americas","USA","San Francisco",90),
    ("Americas","Canada","Toronto",70),
    ("Americas","Canada","Vancouver",55),
    ("EMEA","UK","London",130),
    ("EMEA","UK","Manchester",60),
    ("EMEA","Germany","Berlin",85),
    ("EMEA","Germany","Munich",65),
    ("APAC","Japan","Tokyo",140),
    ("APAC","Japan","Osaka",75),
    ("APAC","Australia","Sydney",95),
    ("APAC","Australia","Melbourne",80),
]

for i, (l1, l2, l3, v) in enumerate(rows):
    r = i + 2  # start at row 2 (Excel is 1-based)
    cells.get(r - 1, 0).putValue(l1)
    cells.get(r - 1, 1).putValue(l2)
    cells.get(r - 1, 2).putValue(l3)
    cells.get(r - 1, 3).putValue(v)

# Make it pretty (optional)
data_range = cells.createRange(0, 0, len(rows) + 1, 4)
data_range.setOutlineBorders(CellBorderType.THIN, Color.getSilver())
ws.autoFitColumns()

# 2) Add Treemap chart
chart_idx = ws.getCharts().add(ChartType.TREEMAP, 1, 6, 28, 15)  # (tRow, lCol, bRow, rCol)
chart = ws.getCharts().get(chart_idx)

chart.getTitle().setText("Sales by Region / Country / City")
chart.getTitle().getFont().setBold(True)

# Values (D2:D13). 'True' => data by columns.
chart.getNSeries().add(f"Data!D2:D{len(rows) + 1}", True)

# Multi-level categories across A..C (A2:C13)
chart.getNSeries().setCategoryData(f"Data!A2:C{len(rows) + 1}")

# Multi-level labels on the category axis
chart.getCategoryAxis().setHasMultiLevelLabels(True)

# Show values & category names (optional)
chart.getNSeries().get(0).getDataLabels().setShowValue(True)
chart.getNSeries().get(0).getDataLabels().setShowCategoryName(True)

# Legend (optional)
chart.getLegend().setPosition(LegendPositionType.RIGHT)

# 3) Save
wb.save("Treemap.xlsx")
print(f"Tree chart created successfully")
jpype.shutdownJVM()
