//date: 2024-02-20T17:01:56Z
//url: https://api.github.com/gists/d268f6d5de604f6a2be61923d02007a5
//owner: https://api.github.com/users/MariannaHon

package org.example;

import org.telegram.telegrambots.meta.TelegramBotsApi;
import org.telegram.telegrambots.updatesreceivers.DefaultBotSession;

import javax.validation.groups.Default;

public class Main {
    public static void main(String[] args) throws Exception {
        var api = new TelegramBotsApi(DefaultBotSession.class);
        api.registerBot(new MyBot());
    }
}