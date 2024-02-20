//date: 2024-02-20T16:54:20Z
//url: https://api.github.com/gists/1634a88be9d297432c452d51523b389f
//owner: https://api.github.com/users/alexkravets00

package org.example;

import org.telegram.telegrambots.meta.TelegramBotsApi;
import org.telegram.telegrambots.updatesreceivers.DefaultBotSession;

public class Main {
    public static void main(String[] args) throws Exception {
        var api = new TelegramBotsApi(DefaultBotSession.class);
        api.registerBot(new MyBot());
    }
}

//aleks1207bot
//6867742594:AAFWF0rnpRjk9Y5TWd9PNx8O5PQXB7ViZSA