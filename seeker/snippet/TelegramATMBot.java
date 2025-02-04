//date: 2025-02-04T17:05:45Z
//url: https://api.github.com/gists/02981d86fcf5e6614c0ebf917a44949a
//owner: https://api.github.com/users/yadnyeshkolte

import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.meta.api.methods.send.SendMessage;
import org.telegram.telegrambots.meta.api.objects.Update;
import java.util.ArrayList;
import java.util.Random;

public class TelegramATMBot extends TelegramLongPollingBot {
    private static final String BOT_TOKEN = "**********"
    private static final String BOT_USERNAME = "your_bot_username";

    // Account Management Variables
    private double accountBalance = 15000.0;
    private List<Double> transactions = new ArrayList<>();
    private int[] denominations = {200, 100, 50};
    private int[] notesLeft = {5, 10, 10};
    private boolean isAuthorized = false;

    @Override
    public void onUpdateReceived(Update update) {
        if (update.hasMessage() && update.getMessage().hasText()) {
            String messageText = update.getMessage().getText();
            long chatId = update.getMessage().getChatId();

            switch (messageText) {
                case "/start":
                    sendWelcomeMessage(chatId);
                    break;
                case "/login":
                    handleLogin(chatId);
                    break;
                case "/withdraw":
                    handleWithdraw(chatId, update);
                    break;
                case "/deposit":
                    handleDeposit(chatId, update);
                    break;
                case "/balance":
                    checkBalance(chatId);
                    break;
                case "/transactions":
                    viewTransactions(chatId);
                    break;
            }
        }
    }

    private void sendWelcomeMessage(long chatId) {
        String welcome = "Welcome to ATM Bot!\n" +
                         "Available Commands:\n" +
                         "/login - Authenticate\n" +
                         "/withdraw - Withdraw Money\n" +
                         "/deposit - Deposit Money\n" +
                         "/balance - Check Balance\n" +
                         "/transactions - View History";
        sendMessage(chatId, welcome);
    }

    private void handleLogin(long chatId) {
        // Generate OTP
        int otp = generateOTP();
        sendMessage(chatId, "Your OTP is: " + otp);
        // Implement OTP verification mechanism
    }

    private int generateOTP() {
        return new Random().nextInt(9000) + 1000;
    }

    private void handleWithdraw(long chatId, Update update) {
        if (!isAuthorized) {
            sendMessage(chatId, "Please login first!");
            return;
        }

        // Implement amount input and validation
        double amount = extractAmount(update);
        
        if (amount % 100 != 0) {
            sendMessage(chatId, "Invalid amount. Must be multiple of 100.");
            return;
        }

        if (amount > accountBalance) {
            sendMessage(chatId, "Insufficient funds!");
            return;
        }

        // Process withdrawal
        processWithdrawal(chatId, amount);
    }

    private void processWithdrawal(long chatId, double amount) {
        accountBalance -= amount;
        transactions.add(-amount);
        
        // Note dispensing logic
        double remainingAmount = amount;
        for (int i = 0; i < denominations.length; i++) {
            while (notesLeft[i] > 0 && remainingAmount >= denominations[i]) {
                remainingAmount -= denominations[i];
                notesLeft[i]--;
            }
        }

        sendMessage(chatId, "Withdrawn: " + amount + 
                    "\nRemaining Balance: " + accountBalance);
    }

    private void handleDeposit(long chatId, Update update) {
        if (!isAuthorized) {
            sendMessage(chatId, "Please login first!");
            return;
        }

        double amount = extractAmount(update);
        
        if (amount % 100 != 0) {
            sendMessage(chatId, "Invalid amount. Must be multiple of 100.");
            return;
        }

        accountBalance += amount;
        transactions.add(amount);
        sendMessage(chatId, "Deposited: " + amount + 
                    "\nNew Balance: " + accountBalance);
    }

    private void checkBalance(long chatId) {
        if (!isAuthorized) {
            sendMessage(chatId, "Please login first!");
            return;
        }
        sendMessage(chatId, "Current Balance: " + accountBalance);
    }

    private void viewTransactions(long chatId) {
        if (!isAuthorized) {
            sendMessage(chatId, "Please login first!");
            return;
        }
        
        StringBuilder transactionHistory = new StringBuilder("Transaction History:\n");
        for (int i = 0; i < transactions.size(); i++) {
            transactionHistory.append("#")
                              .append(i + 1)
                              .append(": ")
                              .append(transactions.get(i))
                              .append("\n");
        }
        sendMessage(chatId, transactionHistory.toString());
    }

    private void sendMessage(long chatId, String text) {
        SendMessage message = new SendMessage();
        message.setChatId(String.valueOf(chatId));
        message.setText(text);
        try {
            execute(message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public String getBotUsername() {
        return BOT_USERNAME;
    }

    @Override
    public String getBotToken() {
        return BOT_TOKEN;
    }
}  }
}