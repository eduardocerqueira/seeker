//date: 2021-09-17T17:01:56Z
//url: https://api.github.com/gists/1fc9736e73a19ff29d59ae21d7fff36e
//owner: https://api.github.com/users/aalbagarcia

package com.cursodegit.twitterFeed;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Base64;
import java.util.Map;
import com.fasterxml.jackson.databind.*;

public class MainClass {

	public static void main(String[] args) throws Exception {
		System.out.println("¡Hola, caracola!");
		
		try {
			Map<String, String> accessTokenMap = getAccessToken();
			
	        System.out.println(accessTokenMap.get("access_token"));
		} catch(Exception e) {
			System.out.println(e.getMessage());
			throw e;
		}
	}
	
	public static Map<String, String> getAccessToken() throws IOException, InterruptedException {
		
        String oAuthConsumerKey = "yv4UDBXKGzuRf58FDzq6O8YVd";
        String oAuthConsumerSecret = "pyWrYPTBHLleabhJ0diG8UIJNlpJDDxz56taz80kVvwvfRYlw8";
        String aux = oAuthConsumerKey + ":" + oAuthConsumerSecret;
		String consumerInfo = Base64.getEncoder().encodeToString(aux.getBytes());;
		
		System.out.println(aux);
		System.out.println(consumerInfo);
		
		// Get the token
		HttpClient client = HttpClient.newHttpClient();
		HttpRequest request = HttpRequest.newBuilder(
		       URI.create("https://api.twitter.com/oauth2/token"))
		   .header("Authorization", "Basic " +  consumerInfo)
		   .header("Content-Type", "application/x-www-form-urlencoded;charset=UTF-8")
		   .POST(HttpRequest.BodyPublishers.ofString("grant_type=client_credentials"))
		   .build();

        HttpResponse<String> response = client.send(request,
                HttpResponse.BodyHandlers.ofString());

		String body = response.body();
	    Map<String, String> map = new ObjectMapper().readValue(body, Map.class);

		
        return map;
	}

}
