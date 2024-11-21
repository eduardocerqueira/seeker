//date: 2024-11-21T17:10:26Z
//url: https://api.github.com/gists/2b36d690b951e3bc20f38f0d586abc0d
//owner: https://api.github.com/users/sts-developer

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class Main {
    public static void main(String[] args) throws Exception {
        // Define the URL with the query parameter
        String url = "https://testapi.taxbandits.com/v1.7.3/form1099B/Validate?SubmissionId=d259edc3-b59a-4771-926d-1f68269a5473";
        URL obj = new URL(url);
        HttpURLConnection con = (HttpURLConnection) obj.openConnection();

        // Set the request method to GET
        con.setRequestMethod("GET");

        // Set request headers (e.g., Authorization, Content-Type)
        con.setRequestProperty("Authorization", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJlYzE0NTIxYjMxNGY0N2RhOTc5ODMzYjVlZjkxNDU5ZSIsImV4cCI6MTcyMTIwNjk3NywiaWF0IjoxNzIxMjAzMzc3LCJpc3MiOiJodHRwczovL3Rlc3RvYXV0aC5leHByZXNzYXV0aC5uZXQvdjIvIiwic3ViIjoiYTQyMWE2MWUzOWUyY2U3ZSJ9.YjQP2gkoecj6tqyXJRcJ8LeeAsP0zwkQYk7iD0QHKW8");
        con.setRequestProperty("Content-Type", "application/json");

        // Get the response code and handle response
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

        // Print the response from the API
        System.out.println(response.toString());
    }
}