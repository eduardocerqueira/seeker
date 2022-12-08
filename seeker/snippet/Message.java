//date: 2022-12-08T16:59:51Z
//url: https://api.github.com/gists/59060d8a567994f4f60bc34e43b0ddd4
//owner: https://api.github.com/users/triangle1984

package com.example.websocketclient;

import lombok.Data;


/**
 * @author Andrey Markov
 * @version 1.0 27.06.2021
 */
@Data
public class Message {
    String from;
    String to;
    String text;

    public Message() {

    }

    public Message(String from) {
        this.from = from;
    }

    public Message(String from, String text, String to) {
        this.from = from;
        this.to = to;
        this.text = text;
    }
}
