//date: 2021-11-12T17:10:37Z
//url: https://api.github.com/gists/39741079d47b8cd56a9f4b709ec99110
//owner: https://api.github.com/users/ValeryVerkhoturov

package com;

//import com.rogurea.GameLoop;
//import com.rogurea.gamemap.Dungeon;
//import com.rogurea.gamemap.Floor;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpRequest.Builder;
import java.net.http.HttpResponse.BodyHandlers;
import java.util.HashMap;
import java.util.concurrent.Semaphore;

class RogueraSpring {
    static HttpRequest getRequest;
    static HttpRequest postRequest;
    static HttpRequest putRequest;
    static int id = 598;
    static String token = "34E11x12z96G11e";
    static String session_id = "2124450348";
    public static void main(String[] args) throws URISyntaxException, IOException, InterruptedException {
//        System.out.println(createNewUser("Сосунок"));
        // 598,34E11x12z96G11e

//        String[] s = map.split(",");
//        Dungeon.player.getPlayerData().setPlayerID(Integer.parseInt(s[0]));  598 id
//        Dungeon.player.getPlayerData().setToken(s[1]); 34E11x12z96G11e token

    //    System.out.println(getUser(id)); //{"id":598,"nickname":"Сосунок","gameSessionList":[]}

//        System.out.println(createGameSession());  2124450348
        updateGameSession();
//        finalizeGameSession();


    }

    public static String getUser(int playerID) throws URISyntaxException, IOException, InterruptedException {
        getRequest = HttpRequest.newBuilder().uri(new URI("http://roguera.tms-studio.ru/users?id=" + playerID)).GET().build();
        HttpResponse<String> response = HttpClient.newHttpClient().send(getRequest, BodyHandlers.ofString());
        HttpHeaders responseHeaders = response.headers();
        return ((String)response.body());
    }

    public static String createNewUser(String nickName) throws URISyntaxException, IOException, InterruptedException {
        postRequest = HttpRequest.newBuilder().uri(new URI("http://roguera.tms-studio.ru/users?userNickName=" + nickName)).POST(BodyPublishers.ofString(nickName)).build();
        HttpResponse<String> response = HttpClient.newHttpClient().send(postRequest, BodyHandlers.ofString());
        return ((String)response.body()).replace("\"", "").replace("[", "").replace("]", "");
    }

    public static int createGameSession() throws URISyntaxException, IOException, InterruptedException {
        Builder var10000 = HttpRequest.newBuilder();
        int var10003 = id;
        postRequest = var10000.uri(new URI("http://roguera.tms-studio.ru/gsessions?userId=" + var10003 + "&userToken=" + token)).POST(BodyPublishers.noBody()).build();
        HttpResponse<String> response = HttpClient.newHttpClient().send(postRequest, BodyHandlers.ofString());
        return Integer.parseInt((String)response.body());
    }

    public static void updateGameSession() throws URISyntaxException, IOException, InterruptedException {
        var values = new HashMap<String, String>() {
            {
                this.put("\"kills\"", String.valueOf(0));
                this.put("\"money_earned\"", "1");
                this.put("\"last_floor\"", "1");
                this.put("\"last_room\"", "1");
                this.put("\"score_earned\"", " 2147483647");
                this.put("\"items\"", "15");
                this.put("\"id_session\"", session_id);
            }
        };
        Builder var10000 = HttpRequest.newBuilder();
        int var10003 = id;
        putRequest = var10000.uri(new URI("http://roguera.tms-studio.ru/gsessions/update?userId=" + var10003 + "&userToken=" + token)).headers(new String[]{"Content-Type", "application/json"}).PUT(BodyPublishers.ofString(values.toString().replace("=", ":"))).build();
        HttpResponse<String> response = HttpClient.newHttpClient().send(putRequest, BodyHandlers.ofString());
    }

    public static void finalizeGameSession() throws URISyntaxException, IOException, InterruptedException {
        putRequest = HttpRequest.newBuilder().uri(new URI("http://roguera.tms-studio.ru/gsessions/finalize?id=" + session_id)).PUT(BodyPublishers.noBody()).build();
        HttpResponse<String> response = HttpClient.newHttpClient().send(putRequest, BodyHandlers.ofString());
    }
}

