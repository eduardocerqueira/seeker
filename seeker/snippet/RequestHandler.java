//date: 2022-11-10T17:02:35Z
//url: https://api.github.com/gists/6d423d5158d9e7db970896472f5cd84b
//owner: https://api.github.com/users/darkbeast0106

package hu.petrik.peoplerestclientkonzol;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

public final class RequestHandler {
    private RequestHandler() {}

    public static Response get(String url) throws IOException {
        HttpURLConnection connection = setupConnection(url);

        connection.setRequestMethod("GET");

        return getResponse(connection);
    }
    
    public static Response post(String url, String data) throws IOException {
        HttpURLConnection connection = setupConnection(url);

        connection.setRequestMethod("POST");
        
        connection.setDoOutput(true);
        OutputStream os = connection.getOutputStream();
        BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(os));
        writer.write(data);
        writer.flush();
        writer.close();
        os.close();

        return getResponse(connection);
    }

    private static HttpURLConnection setupConnection(String url) throws IOException {
        URL urlObj = new URL(url);
        HttpURLConnection connection = (HttpURLConnection) urlObj.openConnection();
        connection.setConnectTimeout(10000);
        connection.setReadTimeout(10000);
        connection.setRequestProperty("Accept", "application/json");
        return connection;
    }

    private static Response getResponse(HttpURLConnection connection) throws IOException {
        int responseCode = connection.getResponseCode();
        InputStream is = connection.getInputStream();
        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        StringBuilder builder = new StringBuilder();
        String line = br.readLine();
        while (line != null) {
            builder.append(line).append(System.lineSeparator());
            line = br.readLine();
        }
        br.close();
        is.close();
        String content = builder.toString().trim();
        return new Response(responseCode, content);
    }
}
