#date: 2024-04-22T16:59:18Z
#url: https://api.github.com/gists/aa3c15d1e8cbb305910f87ce02463dec
#owner: https://api.github.com/users/aspose-com-gists

# This code sample demonstartes how to convert a Visio file to an XAML format in Python.
import aspose.diagram
from aspose.diagram import *

# Load an existing Visio Diagram
diagram = Diagram("sample.vsdx")

# Save diagram in the XAML format
diagram.save("Visio_Converted.xaml", SaveFileFormat.XAML)