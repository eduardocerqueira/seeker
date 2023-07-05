#date: 2023-07-05T16:51:22Z
#url: https://api.github.com/gists/b3e60e3e2d526b7bcb2cdf20e630dbc0
#owner: https://api.github.com/users/aspose-com-kb

import jpype
import asposecells
jpype.startJVM()
from asposecells.api import License, Workbook,\
     PageLayoutAlignmentType, PageLayoutAlignmentType

# Instantiate the license
license = License()
license.setLicense("Aspose.Total.lic")

# Load the workbook
wb = Workbook("ExcelWithPieChart.xlsx")

# Access the worksheet
ws = wb.getWorksheets().get(0)

# Access the chart
chart = ws.getCharts().get(0)

# Convert the chart to PDF
chart.toPdf("ChartToPdf.pdf",10,10,\
            PageLayoutAlignmentType.RIGHT,PageLayoutAlignmentType.BOTTOM)

print("Chart Converted to PDF Successfully")

jpype.shutdownJVM()
