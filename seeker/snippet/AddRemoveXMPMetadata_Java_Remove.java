//date: 2021-10-15T17:01:52Z
//url: https://api.github.com/gists/b3962117fe0b2ced727777f87643bbb7
//owner: https://api.github.com/users/conholdate-gists

// This code example demonstrates how to remove the XMP metadata package from a GIF image.
// Create an instance of the Metadata class
Metadata metadata = new Metadata("C:\\Files\\xmp_output.gif");

// Get root packages
IXmp root = (IXmp)metadata.getRootPackage();

// Set package to null
root.setXmpPackage(null);

// Save image
metadata.save("C:\\Files\\xmp_output_Removed.gif");