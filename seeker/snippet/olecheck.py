#date: 2026-01-29T17:21:28Z
#url: https://api.github.com/gists/e847d2f0ccf8257f8934004fa0206ab3
#owner: https://api.github.com/users/emdnaia

# script to scan MS Office files, looking for "Shell.Explorer" OLE objects which could match CVE-2026-21509
# using oletools - https://github.com/decalage2/oletools
# Philippe Lagadec 2026-01-28

# NOTES:
# According to the MS advisory https://msrc.microsoft.com/update-guide/vulnerability/CVE-2026-21509
# the CVE-2026-21509 vulnerability is related to CLSID "EAB22AC3-30C1-11CF-A7EB-0000C05BAE0B",
# corresponding to the "Shell.Explorer" COM object, which can be used to open the legacy
# Internet Explorer engine (aka Trident/MSHTML) from any application.
# So to exploit CVE2026-21509 from a MS Office document, one could use either an OLE object
# of type "Shell.Explorer", or use an external relationship with a special URL that would
# trigger the use of the Internet Explorer engine, as it was the case for CVE-2021-40444
# with "mhtml:" URLs.
# This script simply tries to identify both cases

# LICENSE:

# olecheck is copyright (c) 2026, Philippe Lagadec (http://www.decalage.info)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import olefile
from oletools import oleobj, ooxml, ftguess, rtfobj

filename = sys.argv[1]
ftg = ftguess.FileTypeGuesser(filename)

if ftg.is_openxml():
    print(f"This is an OpenXML document: {ftg.ftype.longname}")
    for partname in ftg.zipfile.namelist():
        if "oleObject" in partname:
            # print(f"OLE object: {partname}")
            with ftg.zipfile.open(partname) as subfile:
                ole = olefile.OleFileIO(subfile)
                clsid = ole.root.clsid
                print(f"OLE object '{partname}' with CLSID {clsid}")
                if clsid == "EAB22AC3-30C1-11CF-A7EB-0000C05BAE0B":
                    print("=> This *may* be related to CVE-2026-21509")
        elif partname.endswith(".xml"):
            data = ftg.zipfile.read(partname)
            if b"EAB22AC3-30C1-11CF-A7EB-0000C05BAE0B" in data:
                print(f"Part '{partname}' contains the CLSID EAB22AC3-30C1-11CF-A7EB-0000C05BAE0B")
                print("=> This *may* be related to CVE-2026-21509")
    # Also look for external relationships with URLs that do not start with "http"
    # (for example "mhtml:" URLs may launch the IE browser engine, in the case of CVE-2021-40444)
    xml_parser = ooxml.XmlParser(filename)
    for relationship, target in oleobj.find_external_relationships(xml_parser):
        print(f"Found relationship '{relationship}' with external link {target}")
        if not target.startswith('http'):
            print("=> This URL is worth checking")
elif ftg.is_ole():
    print(f"This is an OLE/CFB document: {ftg.ftype.longname}")
    ole = ftg.olefile
    for part in ole.listdir(streams=True, storages=True):
        clsid = ole.getclsid(part)
        if clsid != "":
            partname = '/'.join(part)
            print(f"OLE object '{partname}' with CLSID {clsid}")
            if clsid == "EAB22AC3-30C1-11CF-A7EB-0000C05BAE0B":
                print("=> This *may* be related to CVE-2026-21509")
elif issubclass(ftg.ftype, ftguess.FType_RTF):
    print(f"This is an RTF document: {ftg.ftype.longname}")
    data = open(filename, 'rb').read()
    rtfp = rtfobj.RtfObjParser(data)
    rtfp.parse()
    for obj in rtfp.objects:
        if obj.is_ole:
            print(f"OLE object offset={obj.start} classname={obj.class_name} CLSID={obj.clsid}")
            if obj.clsid == "EAB22AC3-30C1-11CF-A7EB-0000C05BAE0B":
                print("=> This *may* be related to CVE-2026-21509")
else:
    print("This is not a supported MS Office file.")