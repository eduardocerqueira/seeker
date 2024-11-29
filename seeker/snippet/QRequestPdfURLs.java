//date: 2024-11-29T16:59:50Z
//url: https://api.github.com/gists/c05bdbfccaf99da563cbaff4c70eacbc
//owner: https://api.github.com/users/sts-developer

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class Main {
    public static void main(String[] args) throws Exception {
        String url = "https://testapi.taxbandits.com/v1.7.3/Form1099Q/RequestPdfURLs";
        URL obj = new URL(url);
        HttpURLConnection con = (HttpURLConnection) obj.openConnection();

        // Setting basic request headers and method
        con.setRequestMethod("POST");
        con.setRequestProperty("Authorization", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlYzE0NTIxYjMxNGY0N2RhOTc5ODMzYjVlZjkxNDU5ZSIsImV4cCI6MTcyMTIwNjk3NywiaWF0IjoxNzIxMjAzMzc3LCJpc3MiOiJodHRwczovL3Rlc3RvYXV0aC5leHByZXNzYXV0aC5uZXQvdjIvIiwic3ViIjoiYTQyMWE2MWUzOWUyY2U3ZSJ9.YjQP2gkoecj6tqyXJRcJ8LeeAsP0zwkQYk7iD0QHKW8");
        con.setRequestProperty("Content-Type", "application/json");

        // Enable output and set the request body
        con.setDoOutput(true);
        String jsonInputString = "{\n  \"SubmissionId\": \"81b77217-fb5a-4315-b76e-bbb805676a38\",\n  \"RecordIds\": [\n    {\n      \"RecordId\": \"ea13cc94-d25f-4c2f-aef2-a97b3eb59cf1\"\n    }\n  ],\n  \"Customization\": {\n    \"TINMaskType\": \"Both\"\n  }\n}";
        try (DataOutputStream wr = new DataOutputStream(con.getOutputStream())) {
            byte[] input = jsonInputString.getBytes("utf-8");
            wr.write(input, 0, input.length);
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