//date: 2022-05-02T17:20:15Z
//url: https://api.github.com/gists/b5436263eb708baafc34e1189f6a6e29
//owner: https://api.github.com/users/gabriel-antunes-wk

import java.net.HttpURLConnection;
import java.net.URL;
import java.net.MalformedURLException;
import java.net.ProtocolException;
import java.io.IOException;

public class ExemploBearerToken {
  public static HttpURLConnection conexaoComBearerToken(String url, String bearerToken) throws MalformedURLException, IOException, ProtocolException {
    HttpURLConnection conexao = (HttpURLConnection) new URL(url).openConnection();

    conexao.setRequestProperty("Authorization", "Bearer " + bearerToken);
    conexao.setRequestProperty("Content-Type", "application/json");
    conexao.setRequestMethod("GET");

    return conexao;
  }
}