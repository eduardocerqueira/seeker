//date: 2021-10-15T17:01:52Z
//url: https://api.github.com/gists/b3962117fe0b2ced727777f87643bbb7
//owner: https://api.github.com/users/conholdate-gists

// This code example demonstrates how to read all the properties defined in the custom XMP package
// Create an instance of the Metadata class
Metadata metadata = new Metadata("C:\\Files\\xmp_output.gif");

// Get root packages
IXmp root = (IXmp)metadata.getRootPackage();
if (root.getXmpPackage() != null)
{
  // Get Xmp pakages
  XmpPackage[] packages = root.getXmpPackage().getPackages();
  
  // Show Package details
  for (XmpPackage pkg : packages )
  {
    System.out.println(pkg.getNamespaceUri());
    System.out.println(pkg.getPrefix());

    for(String keys : pkg.getKeys())
    {
      MetadataProperty property = pkg.findProperties(new WithNameSpecification(keys)).get_Item(0);
      System.out.println(property.getName() + " : " + property.getValue());
    }
  }
}