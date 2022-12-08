//date: 2022-12-08T16:59:51Z
//url: https://api.github.com/gists/59060d8a567994f4f60bc34e43b0ddd4
//owner: https://api.github.com/users/triangle1984

package com.example.websocketclient;

import lombok.SneakyThrows;
import org.springframework.messaging.simp.stomp.StompCommand;
import org.springframework.messaging.simp.stomp.StompHeaders;
import org.springframework.messaging.simp.stomp.StompSession;
import org.springframework.messaging.simp.stomp.StompSessionHandler;

import java.lang.reflect.Type;

/**
 * @author Andrey Markov
 * @version 1.0 04.07.2021
 */
public class MyStompSessionHandler implements StompSessionHandler {
    private static final org.apache.logging.log4j.Logger log = org.apache.logging.log4j.LogManager.getLogger(MyStompSessionHandler.class);

    @Override
    public void afterConnected(StompSession session, StompHeaders stompHeaders) {

    }

    @SneakyThrows
    @Override
    public void handleException(StompSession stompSession, StompCommand stompCommand, StompHeaders stompHeaders, byte[] bytes, Throwable throwable) {
        throw throwable;
    }

    @SneakyThrows
    @Override
    public void handleTransportError(StompSession stompSession, Throwable throwable) {
        throw throwable;
    }

    @Override
    public Type getPayloadType(StompHeaders stompHeaders) {
        return Message.class;
    }

    @Override
    public void handleFrame(StompHeaders stompHeaders, Object payload) {
        Message message = (Message) payload;
        System.err.printf("От %s пришло сообщение: %s%n", message.getFrom(), message.getText());
    }
}
