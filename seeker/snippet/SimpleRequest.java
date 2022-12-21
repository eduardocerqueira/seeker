//date: 2022-12-21T16:33:59Z
//url: https://api.github.com/gists/129050159275ac4d48c9d08364751f26
//owner: https://api.github.com/users/ridhofauza

import java.util.*;
import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.stream.Collectors;

public class SimpleRequest {

   public static String getStringFromStream(final InputStream inputStream) {
      String result = null;
      BufferedReader streamReader = null;

      try {
         streamReader = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"));
         result = streamReader.lines().collect(Collectors.joining("\n"));
      } catch(UnsupportedEncodingException e) {
         e.printStackTrace();
         return null;
      } finally {
         try {
            if (streamReader != null) {
               streamReader.close();
            }
            if (inputStream != null) {
               inputStream.close();
            }
         } catch(IOException e) {
            e.printStackTrace();
         }
      }

      return result;
   }

   public static String simpleParseArrayProperty(String json, final String porpertyName) {
      if (json == null) return null;
      int lastIndex = json.lastIndexOf(String.format("\"%s\"", porpertyName));
      json = json.substring(lastIndex);
      lastIndex = json.lastIndexOf("[");
      json = json.substring(lastIndex+1);
      return json = json
                     .replaceAll("[\\]}\"]", "")
                     .replaceAll("\\,", ",")
                     .trim();
   }

   public static void main(String[] args) {
      // https://api.chucknorris.io/jokes/random
      System.setProperty("http.agent", "Chrome");
      try {
         URL url = new URL("https://api.chucknorris.io/jokes/random");
         try {
            URLConnection connection = url.openConnection();
            InputStream inputStream = connection.getInputStream();
            System.out.println(simpleParseArrayProperty(getStringFromStream(inputStream), "value"));
         } catch(IOException e) {
            e.printStackTrace();
         }
      } catch(MalformedURLException e) {
         e.printStackTrace();
      }
   }

}