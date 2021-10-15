//date: 2021-10-15T17:01:52Z
//url: https://api.github.com/gists/b3962117fe0b2ced727777f87643bbb7
//owner: https://api.github.com/users/conholdate-gists

// This code example demonstrates how to create and add a custom XMP metadata package to a GIF image.
// Create an instance of the Metadata class
Metadata metadata = new Metadata("C:\\Files\\xmp.gif");

// Get root packages
IXmp root = (IXmp)metadata.getRootPackage();

// Create Xmp Packet Wrapper
XmpPacketWrapper packet = new XmpPacketWrapper();

// Define custom package
XmpPackage custom = new XmpPackage("gd", "https://groupdocs.com");
custom.set("gd:Copyright", "Copyright (C) 2021 GroupDocs. All Rights Reserved.");
custom.set("gd:CreationDate", new Date().toString());
custom.set("gd:Company", XmpArray.from(new String[] { "Aspose", "GroupDocs" }, XmpArrayType.Ordered));

// Add custom package to Xmp Packet Wrapper
packet.addPackage(custom);

// Update XmpPackage
root.setXmpPackage(packet);

// Save the file
metadata.save("C:\\Files\\xmp_output.gif");