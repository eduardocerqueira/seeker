//date: 2023-02-15T16:54:58Z
//url: https://api.github.com/gists/7c4b634bedfee0ea846059eb9f9a7d82
//owner: https://api.github.com/users/VenkataA

/*
   * This Java Quick Start uses the SOAP mode and contains the following JAR files
   * in the class path:
    * 1. adobe-distiller-client.jar
    * 2. adobe-livecycle-client.jar
    * 3. adobe-usermanager-client.jar
    * 4. adobe-utilities.jar
    * 5. jboss-client.jar (use a different JAR file if the forms server is not deployed
    * on JBoss)
    * 6. activation.jar (required for SOAP mode)
    * 7. axis.jar (required for SOAP mode)
    * 8. commons-codec-1.3.jar (required for SOAP mode)
    * 9.  commons-collections-3.1.jar  (required for SOAP mode)
    * 10. commons-discovery.jar (required for SOAP mode)
    * 11. commons-logging.jar (required for SOAP mode)
    * 12. dom3-xml-apis-2.5.0.jar (required for SOAP mode)
    * 13. jaxen-1.1-beta-9.jar (required for SOAP mode)
    * 14. jaxrpc.jar (required for SOAP mode)
    * 15. log4j.jar (required for SOAP mode)
    * 16. mail.jar (required for SOAP mode)
    * 17. saaj.jar (required for SOAP mode)
    * 18. wsdl4j.jar (required for SOAP mode)
    * 19. xalan.jar (required for SOAP mode)
    * 20. xbean.jar (required for SOAP mode)
    * 21. xercesImpl.jar (required for SOAP mode)
    *
    * These JAR files are located in the following path:
    * <install directory>/sdk/client-libs/common
    *
    * The adobe-utilities.jar file is located in the following path:
    * <install directory>/sdk/client-libs/jboss
    *
    * The jboss-client.jar file is located in the following path:
    * <install directory>/jboss/bin/client
    *
    * SOAP required JAR files are located in the following path:
    * <install directory>/sdk/client-libs/thirdparty
    *
    * If you want to invoke a remote forms server instance and there is a
    * you have to include these additional JAR files
    *
    * For information about the SOAP
    * mode, see "Setting connection properties" in Programming
    * with AEM Forms
    */
import java.io.File;
import java.io.FileInputStream;
import java.util.Properties;
import com.adobe.idp.Document;
import com.adobe.idp.dsc.clientsdk.ServiceClientFactory;
import com.adobe.idp.dsc.clientsdk.ServiceClientFactoryProperties;
import com.adobe.livecycle.distiller.client.DistillerServiceClient;
import com.adobe.livecycle.generatepdf.client.ConversionException;
import com.adobe.livecycle.generatepdf.client.CreatePDFResult;
import com.adobe.livecycle.generatepdf.client.FileFormatNotSupportedException;
import com.adobe.livecycle.generatepdf.client.InvalidParameterException;
public class PStoPDF {
	public static void main(String[] args) {
		
		try {
		//Set connection properties required to invoke AEM Forms using SOAP mode
        Properties connectionProps = new Properties();
        connectionProps.setProperty(ServiceClientFactoryProperties.DSC_DEFAULT_SOAP_ENDPOINT, "http://localhost:8080");
        connectionProps.setProperty(ServiceClientFactoryProperties.DSC_TRANSPORT_PROTOCOL,ServiceClientFactoryProperties.DSC_SOAP_PROTOCOL);
        connectionProps.setProperty(ServiceClientFactoryProperties.DSC_SERVER_TYPE, "JBoss");
        connectionProps.setProperty(ServiceClientFactoryProperties.DSC_CREDENTIAL_USERNAME, "administrator");
        connectionProps.setProperty(ServiceClientFactoryProperties.DSC_CREDENTIAL_PASSWORD, "password");
     // Create a ServiceClientFactory instance
        ServiceClientFactory factory = ServiceClientFactory.createInstance(connectionProps);
        //String inputFileName = "C:/delete/11362431_export.ps";
        String folderName = "C:\\delete\\20230213_PS\\ps-test-part2";
        File inputFolder = new File(folderName);
       File[] files = inputFolder.listFiles();
       for (int i=0; i<files.length; i++) {
        //Document createdDocument = converPStoPDF(factory, inputFileName);
    	 Document createdDocument = PStoPDF.converPStoPDF(factory, files[i].getAbsolutePath());
    	
    	
         //Save the PDF file
        createdDocument.copyToFile(new File(folderName, files[i].getName()+".pdf"));
        }
       }
        catch (Exception e) {
            e.printStackTrace();
        }
	}
	static Document converPStoPDF(ServiceClientFactory factory, String inputFileName)
			throws InvalidParameterException, ConversionException, FileFormatNotSupportedException {
		DistillerServiceClient disClient = new DistillerServiceClient(factory );
        // Get a PS file document to convert to a PDF document and populate a com.adobe.idp.Document object
        //String inputFileName = "c:/delete/test_ret.eps";
        //FileInputStream fileInputStream = new FileInputStream(inputFileName);
        //Document inDoc = new Document(fileInputStream);
        File inputFile = new File(inputFileName);
        Document inDoc = new Document(inputFile,false);
        //Set run-time options
        //String adobePDFSettings = "PDFA1b 2005 RGB";
        String adobePDFSettings = "Standard";
         String securitySettings = "No Security";
         //Convert a PS  file into a PDF file
        CreatePDFResult result = new CreatePDFResult();
/*        result = disClient.createPDF(
                inDoc,
                inputFileName,
                     adobePDFSettings,
                securitySettings,
                null,
                null
            );
*/
        result = disClient.createPDF2(inDoc, "ps", adobePDFSettings, securitySettings, null, null);
         //Get the newly created document
         Document createdDocument = result.getCreatedDocument();
		return createdDocument;
	}
}