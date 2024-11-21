//date: 2024-11-21T17:06:29Z
//url: https://api.github.com/gists/e67759f96ff9272a2e559622086bc09d
//owner: https://api.github.com/users/sts-developer

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class Main {
    public static void main(String[] args) throws Exception {
        String url = "https://testapi.taxbandits.com/v1.7.3/Form1099B/ValidateForm";
        URL obj = new URL(url);
        HttpURLConnection con = (HttpURLConnection) obj.openConnection();

        // Setting basic post request headers
        con.setRequestMethod("POST");
        con.setRequestProperty("Authorization", "YOUR_AUTH_TOKEN_HERE");
        con.setRequestProperty("Content-Type", "application/json");

        // Enabling POST parameters
        con.setDoOutput(true);

        // JSON payload as a single-line string (Updated JSON)
        String postData = "{"
                + "\"SubmissionManifest\": {"
                + "\"TaxYear\": \"2024\","
                + "\"IRSFilingType\":\"IRIS\","
                + "\"IsFederalFiling\": true,"
                + "\"IsPostal\": true,"
                + "\"IsOnlineAccess\": true,"
                + "\"IsScheduleFiling\": false,"
                + "\"ScheduleFiling\": null"
                + "},"
                + "\"ReturnHeader\": {"
                + "\"Business\": {"
                + "\"BusinessId\": null,"
                + "\"BusinessNm\": \"Snowdaze LLC\","
                + "\"FirstNm\": null,"
                + "\"MiddleNm\": null,"
                + "\"LastNm\": null,"
                + "\"Suffix\": null,"
                + "\"PayerRef\": \"Snow123\","
                + "\"TradeNm\": \"Iceberg Icecreams\","
                + "\"IsEIN\": true,"
                + "\"EINorSSN\": \"71-3787159\","
                + "\"Email\": \"james@sample.com\","
                + "\"ContactNm\": null,"
                + "\"Phone\": \"6634567890\","
                + "\"PhoneExtn\": \"12345\","
                + "\"Fax\": \"6634567890\","
                + "\"BusinessType\": \"ESTE\","
                + "\"SigningAuthority\": null,"
                + "\"KindOfEmployer\": \"FederalGovt\","
                + "\"KindOfPayer\": \"REGULAR941\","
                + "\"IsBusinessTerminated\": true,"
                + "\"IsForeign\": false,"
                + "\"USAddress\": {"
                + "\"Address1\": \"3576 AIRPORT WAY\","
                + "\"Address2\": \"UNIT 9\","
                + "\"City\": \"FAIRBANKS\","
                + "\"State\": \"AK\","
                + "\"ZipCd\": \"99709\""
                + "},"
                + "\"ForeignAddress\": null"
                + "}"
                + "},"
                + "\"ReturnData\": ["
                + "{"
                + "\"SequenceId\": \"1\","
                + "\"IsPostal\": true,"
                + "\"IsOnlineAccess\": true,"
                + "\"IsForced\": true,"
                + "\"Recipient\": {"
                + "\"RecipientId\": null,"
                + "\"TINType\": \"EIN\","
                + "\"TIN\": \"36-3814577\","
                + "\"FirstPayeeNm\": \"Dairy Delights LLC\","
                + "\"SecondPayeeNm\": \"Coco Milk\","
                + "\"FirstNm\": null,"
                + "\"MiddleNm\": null,"
                + "\"LastNm\": null,"
                + "\"Suffix\": null,"
                + "\"IsForeign\": true,"
                + "\"USAddress\": null,"
                + "\"ForeignAddress\": {"
                + "\"Address1\": \"120 Bremner Blvd\","
                + "\"Address2\": \"Suite 800\","
                + "\"City\": \"Toronto\","
                + "\"ProvinceOrStateNm\": \"Ontario\","
                + "\"Country\": \"CA\","
                + "\"PostalCd\": \"4168682600\""
                + "},"
                + "\"Email\": \"shawn09@sample.com\","
                + "\"Fax\": \"6834567890\","
                + "\"Phone\": \"7634567890\""
                + "},"
                + "\"BFormData\": {"
                + "\"B1aDescrOfProp\": \"RFC\","
                + "\"B1bDateAcquired\": \"07/01/2022\","
                + "\"B1cDateSoldOrDisposed\": \"09/04/2021\","
                + "\"B1dProceeds\": 40.55,"
                + "\"B1eCostOrOtherBasis\": 30.89,"
                + "\"B1fAccruedMktDisc\": 20.11,"
                + "\"B1gWashsaleLossDisallowed\": 4.25,"
                + "\"B2TypeOfGainLoss\": \"ordinary short term\","
                + "\"B3IsProceedsFromCollectibles\": true,"
                + "\"B3IsProceedsFromQOF\": false,"
                + "\"B4FedTaxWH\": 0,"
                + "\"B5IsNonCoveredSecurityNotReported\": false,"
                + "\"B5IsNonCoveredSecurityReported\": false,"
                + "\"B6IsGrossProceeds\": true,"
                + "\"B6IsNetProceeds\": false,"
                + "\"B7IsLossNotAllowedbasedOn1d\": false,"
                + "\"B8PLRealizedOnClosedContract\": 0,"
                + "\"B9PLUnrealizedOnOpenContractPrevTy\": 0,"
                + "\"B10UnrealizedPLOnOpenContractCurTy\": 0,"
                + "\"B11AggPLOnContract\": 0,"
                + "\"B12IsBasisReportedToIRS\": false,"
                + "\"B13Bartering\": 43,"
                + "\"AccountNum\": \"789121\","
                + "\"CUSIPNum\": \"8988932143534\","
                + "\"IsFATCA\": true,"
                + "\"Form8949Code\": \"X\","
                + "\"Is2ndTINnot\": true,"
                + "\"States\": ["
                + "{"
                + "\"StateCd\": \"WV\","
                + "\"StateIdNum\": \"99999999\","
                + "\"StateWH\": 257.94"
                + "},"
                + "{"
                + "\"StateCd\": \"ID\","
                + "\"StateIdNum\": \"999999999\","
                + "\"StateWH\": 15"
                + "}"
                + "]"
                + "}"
                + "}"
                + "]"
                + "}";

        // Writing the POST data to the request
        try (DataOutputStream wr = new DataOutputStream(con.getOutputStream())) {
            wr.writeBytes(postData);
            wr.flush();
        }

        // Reading the response
        int responseCode = con.getResponseCode();
        BufferedReader in;
        if (responseCode >= 200 && responseCode <= 300) {
            in = new BufferedReader(new InputStreamReader(con.getInputStream()));
        } else {
            in = new BufferedReader(new InputStreamReader(con.getErrorStream()));
        }
        String inputLine;
        StringBuilder response = new StringBuilder();
        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();

        // Printing the response
        System.out.println(response.toString());
    }
}