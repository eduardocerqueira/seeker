//date: 2022-12-21T17:03:06Z
//url: https://api.github.com/gists/5fe7e086184a98dc28f6dc7874f427d6
//owner: https://api.github.com/users/ridhofauza

import java.util.*;
import java.io.*;
import java.net.*;

public class GetRequest {
   public static void main(String[] args) {
      try {
         URL url = new URL("https://api.chucknorris.io/jokes/random");
         HttpURLConnection huc = (HttpURLConnection) url.openConnection();
         huc.setRequestMethod("GET");
         huc.setRequestProperty("User-Agent", "Chrome");
         InputStream inputStream = huc.getInputStream();
         BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));

         StringBuilder strBuilder = new StringBuilder();
         String text = "";
         while((text = br.readLine()) != null) {
            strBuilder.append(text);
         }

         System.out.println(strBuilder.toString());
         huc.disconnect();
      } catch(Exception e) {
         System.out.println(e);
      }
   }
}