//date: 2022-12-08T16:59:51Z
//url: https://api.github.com/gists/59060d8a567994f4f60bc34e43b0ddd4
//owner: https://api.github.com/users/triangle1984

package com.example.websocketclient;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.messaging.converter.MappingJackson2MessageConverter;
import org.springframework.messaging.simp.stomp.StompFrameHandler;
import org.springframework.messaging.simp.stomp.StompHeaders;
import org.springframework.messaging.simp.stomp.StompSession;
import org.springframework.messaging.simp.stomp.StompSessionHandler;
import org.springframework.web.socket.WebSocketHttpHeaders;
import org.springframework.web.socket.client.WebSocketClient;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.messaging.WebSocketStompClient;

import java.lang.reflect.Type;
import java.util.Scanner;
import java.util.concurrent.ExecutionException;

@SpringBootApplication
public class WebSocketClientApplication {
    final static String URL = "ws://localhost:8080/ws";
    public static String userId = "t";

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        Scanner sc = new Scanner(System.in);
//        System.out.print("Введите имя пользователя: ");
//        userId = sc.nextLine();
        StompHeaders headers = new StompHeaders();
        headers.set("simpUser", "test");
        WebSocketClient webSocketClient = new StandardWebSocketClient();
        WebSocketStompClient stompClient = new WebSocketStompClient(webSocketClient);
        stompClient.setMessageConverter(new MappingJackson2MessageConverter());
        StompSessionHandler sessionHandler = new MyStompSessionHandler();
        StompSession client = stompClient.connect(URL, sessionHandler, headers).get();
//        while (true) {
//            System.out.println("чо?");
//            String text = sc.nextLine();
            client.send("/app/messages", new Message(userId, "Ебана врот, это сработало", "test"));
            Thread.sleep(8000);
//        }
    }

}
