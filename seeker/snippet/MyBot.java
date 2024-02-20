//date: 2024-02-20T16:54:20Z
//url: https://api.github.com/gists/1634a88be9d297432c452d51523b389f
//owner: https://api.github.com/users/alexkravets00

package org.example;

import net.thauvin.erik.crypto.CryptoPrice;
import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.meta.api.methods.send.SendMessage;
import org.telegram.telegrambots.meta.api.objects.Update;

public class MyBot extends TelegramLongPollingBot {
    public MyBot() {
        super("6867742594:AAFWF0rnpRjk9Y5TWd9PNx8O5PQXB7ViZSA");
    }

    @Override
    public void onUpdateReceived(Update update) {
        var chatId = update.getMessage().getChatId();
        var text = update.getMessage().getText();


        try {
            if (text.equals("/start")) {
                sendMessage(chatId,"Hello!");
            } else if (text.equals("btc")) {
                sendPrice(chatId, "BTC");
            } else if (text.equals("eth")) {
                sendPrice(chatId, "ETH");
            } else if (text.equals("eos")) {
                sendPrice(chatId, "EOS");
            } else if   (text.equals("btc,eth")) {
                var price = CryptoPrice.spotPrice("BTC");
                var price1 = CryptoPrice.spotPrice("ETH");
                sendMessage(chatId, "BTC price: " + price.getAmount().doubleValue() + ", ETH price: " + price1.getAmount().doubleValue());
            }else if   (text.equals("btc,eth,eos")) {
                var price = CryptoPrice.spotPrice("BTC");
                var price1 = CryptoPrice.spotPrice("ETH");
                var price2 = CryptoPrice.spotPrice("EOS");
                sendMessage(chatId,"BTC price: " + price.getAmount().doubleValue() + ", ETH price: " + price1.getAmount().doubleValue() + ", EOS price: " + price2.getAmount().doubleValue());
            } else {
                sendMessage(chatId, "Unknown command!");
            }
        } catch (Exception e) {
            System.out.println("Error!");
        }
    }

    void sendPrice(long chatId, String name) throws Exception {
        var price = CryptoPrice.spotPrice(name);
        sendMessage(chatId, name + " price: " + price.getAmount().doubleValue());
    }

    void sendMessage(long chatId, String text) throws Exception {
        var message = new SendMessage();
        message.setChatId(chatId);
        message.setText(text);
        execute(message);
    }

    @Override
    public String getBotUsername() {
        return "aleks1207bot";
    }
}