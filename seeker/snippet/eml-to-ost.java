//date: 2025-03-03T17:09:59Z
//url: https://api.github.com/gists/a9b78ec14dc41066822013fccdabd67e
//owner: https://api.github.com/users/aspose-com-gists

package com.example;
import com.aspose.email.*;

public class main {
	public static void main(String[] args) {
		// Define the working directory path. 
		String dir = "data";
		try {
		    // Set the path to the EML and OST files
		    String emlFilePath = dir+"sample.eml";
		    String ostFilePath = dir+"output_file.ost";
		    // Load EML file by invoking the load method. 
		    MailMessage eml = MailMessage.load(emlFilePath);
		    // Invoke the fromMailMessage method to convert EML to MapiMessage.
		    MapiMessage mapiMsg = MapiMessage.fromMailMessage(eml);
		    // Call the create method to create an OST file. 
		    PersonalStorage ost = PersonalStorage.create(ostFilePath, FileFormatVersion.Unicode);
		    // Create Inbox folder by invoking the addSubFolder method.
		    ost.getRootFolder().addSubFolder("Inbox");
		    // The getSubFolder method will get the Inbox folder and then add the message by calling the addMessage method. 
		    FolderInfo inbox = ost.getRootFolder().getSubFolder("Inbox");
		    inbox.addMessage(mapiMsg);
		    // Save OST file
		    ost.dispose();
		    System.out.println("EML converted to OST successfully!");
		} catch (Exception e) {
		    System.err.println("Error: " + e.getMessage());
		}
	}
}