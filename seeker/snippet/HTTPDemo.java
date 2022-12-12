//date: 2022-12-12T16:52:13Z
//url: https://api.github.com/gists/00998b3da8cd841e97872c0cfe9119c1
//owner: https://api.github.com/users/Thomas-Hartmann

package demos;
import okhttp3.*;
import java.io.IOException;

public class HTTPDemo {

        public static void main(String[] args) throws IOException {
//               httpGet();
//               httpPost();
            httpPost2();
        }
        private static void httpGet() throws IOException {
            OkHttpClient client = new OkHttpClient().newBuilder()
                    .build();
            Request request = new Request.Builder()
                    .url("https://jsonplaceholder.typicode.com/posts/1")
                    .method("GET", null)
                    .build();
            Response response = client.newCall(request).execute();
            String res = response.body().string();
            System.out.println(res);
        }
        private static void httpPost() throws IOException {
            OkHttpClient client = new OkHttpClient().newBuilder()
                    .build();
            MediaType mediaType = MediaType.parse("application/json");
//            RequestBody body = RequestBody.create(mediaType, "{\r\n    \"title\": \"foo\",\r\n    \"body\": \"bar\",\r\n    \"userId\": 1\r\n}");
            RequestBody body = RequestBody.create("{\r\n    \"title\": \"foo\",\r\n    \"body\": \"bar\",\r\n    \"userId\": 1\r\n}", mediaType);
            Request request = new Request.Builder()
                    .url("https://jsonplaceholder.typicode.com/posts")
                    .method("POST", body)
                    .addHeader("Content-Type", "application/json")
                    .build();
            Response response = client.newCall(request).execute();
            String res = response.body().string();
            System.out.println(res);
        }
    private static void httpPost2() throws IOException {

        // form parameters
        RequestBody formBody = new FormBody.Builder()
                .add("title", "abc")
                .add("body", "123")
                .add("userId", String.valueOf(1))
                .build();

        Request request = new Request.Builder()
                .url("https://jsonplaceholder.typicode.com/posts")
                .addHeader("User-Agent", "OkHttp Bot")
                .addHeader("Content-Type", "application/json")
                .post(formBody)
                .build();
        OkHttpClient client = new OkHttpClient();
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);
            // Get response body
            System.out.println(response.body().string());
        }

    }
}
