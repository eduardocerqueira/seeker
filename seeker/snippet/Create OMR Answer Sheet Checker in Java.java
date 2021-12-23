//date: 2021-12-23T16:36:50Z
//url: https://api.github.com/gists/62b9c5ecd054c010906ff3ca69202b5b
//owner: https://api.github.com/users/aspose-com-kb

import com.aspose.omr.License;
import com.aspose.omr.OmrEngine;
import com.aspose.omr.TemplateProcessor;
import java.io.PrintWriter;

public class OMRAnswerSheetCheckerInJava {      
    public static void main(String[] args) throws Exception { // main method for creating OMR Answer sheet checker

        // Set Aspose.OMR license before creating OMR answer sheet in Java
        License OMRLicense = new License();
        OMRLicense.setLicense("Aspose.OMR.lic");

        // Load the template file into template processor
        OmrEngine OMREngineJava = new OmrEngine();
        TemplateProcessor OMRTemplateProcessorJava = OMREngineJava.getTemplateProcessor("Template.omr");

        // Get CSV format output from the image
        String OutputCSVFromImage = OMRTemplateProcessorJava.recognizeImage("AnswerSheetImageToOMR.png").getCsv();

        try (PrintWriter OMRCSVwriter = new PrintWriter("OMRCSVoutput.txt")) {
        OMRCSVwriter.println(OutputCSVFromImage);
        }
    }
}