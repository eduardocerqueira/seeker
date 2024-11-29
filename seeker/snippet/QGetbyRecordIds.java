//date: 2024-11-29T17:05:55Z
//url: https://api.github.com/gists/2a0dd96de3728c31cfbf2bd83d12ec66
//owner: https://api.github.com/users/sts-developer

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

public class Main {
    public static void main(String[] args) throws Exception {
        String url = "https://testapi.taxbandits.com/v1.7.3/Form1099Q/GetbyRecordIds";
        URL obj = new URL(url);
        HttpURLConnection con = (HttpURLConnection) obj.openConnection();

        // Setting basic request headers and method
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/json");
        con.setRequestProperty("Authorization", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlYzE0NTIxYjMxNGY0N2RhOTc5ODMzYjVlZjkxNDU5ZSIsImV4cCI6MTcyMTIwNjk3NywiaWF0IjoxNzIxMjAzMzc3LCJpc3MiOiJodHRwczovL3Rlc3RvYXV0aC5leHByZXNzYXV0aC5uZXQvdjIvIiwic3ViIjoiYTQyMWE2MWUzOWUyY2U3ZSJ9.YjQP2gkoecj6tqyXJRcJ8LeeAsP0zwkQYk7iD0QHKW8");


        // Enabling input and output streams
        con.setDoOutput(true);

        // JSON payload to be sent in the request body
        String jsonInputString = "{\r\n" +
                "    \"RecordIds\": [\r\n" +
                "        {\r\n" +
                "            \"RecordId\": \"7e2b2902-fc16-4dce-bdfd-747617b35c77\"\r\n" +
                "        },\r\n" +
                "        {\r\n" +
                "            \"RecordId\": \"7e2b2902-fc16-4dce-bdfd-747617b35c77\"\r\n" +
                "        }\r\n" +
                "    ]\r\n" +
                "}";

        // Writing the JSON payload to the output stream
        try (OutputStream os = con.getOutputStream()) {
            byte[] input = jsonInputString.getBytes("utf-8");
            os.write(input, 0, input.length);
        }

        // Reading the response
        BufferedReader in;
        int responseCode = con.getResponseCode();
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