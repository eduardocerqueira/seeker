#date: 2022-10-28T17:10:00Z
#url: https://api.github.com/gists/67102c6f6ef3d8a5b6706cd0f85b6c18
#owner: https://api.github.com/users/aspose-com-kb


import aspose.slides as slides
# The path to source files directory
filePath = "C://Words//"

#Load the license in your application to read the presentation document properties
pptxDocsPropertiesLicense = slides.License()
pptxDocsPropertiesLicense.set_license(filePath + "Conholdate.Total.Product.Family.lic")

# Use the IPresentationInfo object to read the presentation info from the presentation factory
presInfo = slides.PresentationFactory.instance.get_presentation_info(filePath + "NewPresentation.pptx") 

# Fetch the presentation document properties
props = presInfo.read_document_properties()

# Access and display the presentation document properties

print("Subject :"+ props.subject)

print("Title : "+props.title)

print("Author : "+props.author)

print("Comments : "+props.comments)

print("RevisionNumber : "+ str(props.revision_number))

print("CreatedTime :" + props.created_time.strftime('%m/%d/%Y'))

print("Process Completed")