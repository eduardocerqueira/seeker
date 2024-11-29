//date: 2024-11-29T17:12:17Z
//url: https://api.github.com/gists/b065512fb8ab337c0814e5c185b32c39
//owner: https://api.github.com/users/sts-developer

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class Main {
    public static void main(String[] args) throws Exception {
        String url = "https://testapi.taxbandits.com/v1.7.3/Form1099PATR/RequestDraftPdfUrl";
        URL obj = new URL(url);
        HttpURLConnection con = (HttpURLConnection) obj.openConnection();

        // Setting basic request headers and method
        con.setRequestMethod("POST");
        con.setRequestProperty("Authorization", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlYzE0NTIxYjMxNGY0N2RhOTc5ODMzYjVlZjkxNDU5ZSIsImV4cCI6MTcyMTIwNjk3NywiaWF0IjoxNzIxMjAzMzc3LCJpc3MiOiJodHRwczovL3Rlc3RvYXV0aC5leHByZXNzYXV0aC5uZXQvdjIvIiwic3ViIjoiYTQyMWE2MWUzOWUyY2U3ZSJ9.YjQP2gkoecj6tqyXJRcJ8LeeAsP0zwkQYk7iD0QHKW8");
        con.setRequestProperty("Content-Type", "application/json");

        // Enable input and output
        con.setDoOutput(true);
        con.setDoInput(true);

        // JSON request body
        String jsonInputString = "{\r\n  \"TaxYear\": null,\r\n  \"RecordId\": \"cf0a188b-6661-4b57-b04b-ba9ead52a16e\",\r\n  \"Business\": null,\r\n  \"Recipient\": null\r\n}";

        // Writing the JSON body to the connection
        try (DataOutputStream wr = new DataOutputStream(con.getOutputStream())) {
            wr.writeBytes(jsonInputString);
            wr.flush();
        }

        // Reading the response
        int responseCode = con.getResponseCode();
        BufferedReader in;
        if (responseCode >= 200 && responseCode < 300) {
            in = new BufferedReader(new InputStreamReader(con.getInputStream()));
        } else {
            in = new BufferedReader(new InputStreamReader(con.getErrorStream()));
        }
        String inputLine;
        StringBuffer response = new StringBuffer();
        while ((inputLine = in.readLine()) != null) {
            response.append(inputLine);
        }
        in.close();

        // Printing the response
        System.out.println(response.toString());
    }
}