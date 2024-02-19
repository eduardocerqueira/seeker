//date: 2024-02-19T17:07:49Z
//url: https://api.github.com/gists/035c37c6c92676ca6b39b34cd01b0fb2
//owner: https://api.github.com/users/Olga-Chorna

package org.example;

import net.thauvin.erik.crypto.CryptoPrice;
import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.meta.api.methods.send.SendMessage;
import org.telegram.telegrambots.meta.api.methods.send.SendPhoto;
import org.telegram.telegrambots.meta.api.objects.InputFile;
import org.telegram.telegrambots.meta.api.objects.Update;
import org.telegram.telegrambots.meta.exceptions.TelegramApiException;

public class MyBot extends TelegramLongPollingBot {
    public MyBot() {
        super("6624632592:AAFeglhMzsPXQmlTeAzn6PpzgTAS_dXhpn0");
    }
    @Override
    public void onUpdateReceived(Update update) {
        var chatId = update.getMessage().getChatId();
        var text = update.getMessage().getText();

        try {
            if(text.equals("/start")){
                sendMessage(chatId, "Hello");
            } else if (text.equals("btc")) {
                sendPicture(chatId, "bitcoin-btc-logo.png");
                sendPrice(chatId,"BTC");
            } else if (text.equals("eth")) {
                sendPicture(chatId, "ethereum-eth-logo.png");
                sendPrice(chatId,"ETH");
            } else if (text.equals("ltc")) {
                sendPicture(chatId, "litecoin-ltc-logo.png");
                sendPrice(chatId,"LTC");
            } else if (text.equals("btc 100")) {
                sendCalculatedPrice(chatId, "BTC", 100);
            } else if (text.equals("eth 350")) {
                sendCalculatedPrice(chatId, "ETH", 350);
            } else if (text.equals("/all")) {
                String[] cryptocurrencies = {"BTC", "ETH", "LTC"};
              
                for (int i = 0; i < cryptocurrencies.length; i++) {
                    sendPrice(chatId,cryptocurrencies[i]);
                }
              
            } else {
                sendMessage(chatId,"Unknown command!");
            }

        } catch (Exception e) {
            System.out.println("Error");
        }
    }

    void sendPicture(long chatId, String name) throws Exception {
        var photo = getClass().getClassLoader().getResourceAsStream(name);

        var message = new SendPhoto();
        message.setChatId(chatId);
        message.setPhoto(new InputFile(photo, name));
        execute(message);
    }

    void sendMessage(long chatId, String text) throws Exception {
        var message = new SendMessage();
        message.setChatId(chatId);
        message.setText(text);
        execute(message);
    }


    void sendPrice(long chatId, String name) throws Exception {
        var price = CryptoPrice.spotPrice(name);
        sendMessage(chatId, name + " price: " + price.getAmount().doubleValue());
    }

    void sendCalculatedPrice(long chatId, String name, long base) throws Exception {
        var price = CryptoPrice.spotPrice(name);
        var result = base / price.getAmount().doubleValue();
        sendMessage(chatId, "You can buy " + result + " " + name);
    }

    @Override
    public String getBotUsername() {
        return "Black_bot_for_PA_bot";
    }
}
