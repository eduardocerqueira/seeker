//date: 2025-02-04T17:05:45Z
//url: https://api.github.com/gists/02981d86fcf5e6614c0ebf917a44949a
//owner: https://api.github.com/users/yadnyeshkolte

import telebot
import random
import sqlite3

class TelegramATMBot:
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
        self.bot = "**********"
        self.authorized_users = {}
        self.account_balance = 15000
        self.transactions = []
        self.denominations = [200, 100, 50]
        self.notes_left = [5, 10, 10]
        
        # Setup bot message handlers
        self.register_handlers()

    def register_handlers(self):
        @self.bot.message_handler(commands=['start'])
        def send_welcome(message):
            """Display welcome message and available commands"""
            welcome_text = """
            Welcome to ATM Bot! 
            Available Commands:
            /login - Login to ATM
            /withdraw - Withdraw money
            /deposit - Deposit money
            /balance - Check balance
            /transactions - View transaction history
            """
            self.bot.reply_to(message, welcome_text)

    def authenticate(self, user_id):
        """User authentication mechanism"""
        otp = random.randint(1000, 9999)
        # Send OTP via Telegram
        self.bot.send_message(user_id, f"Your OTP is: {otp}")
        return self.verify_otp(user_id, otp)

    def verify_otp(self, user_id, generated_otp):
        """OTP verification logic"""
        # Implement OTP verification mechanism
        pass

    def withdraw_money(self, amount):
        """Money withdrawal logic"""
        if amount % 100 != 0:
            return "Invalid amount. Must be multiple of 100."
        
        if amount > self.account_balance:
            return "Insufficient funds."
        
        # Implement note dispensing logic
        temp_amount = amount
        for i, denomination in enumerate(self.denominations):
            while self.notes_left[i] > 0 and temp_amount >= denomination:
                temp_amount -= denomination
                self.notes_left[i] -= 1
        
        self.account_balance -= amount
        self.transactions.append(amount)
        return f"Withdrawn {amount}. Remaining balance: {self.account_balance}"

    def deposit_money(self, amount):
        """Money deposit logic"""
        if amount % 100 != 0:
            return "Invalid amount. Must be multiple of 100."
        
        self.account_balance += amount
        self.transactions.append(amount)
        return f"Deposited {amount}. New balance: {self.account_balance}"

    def check_balance(self):
        """Balance inquiry"""
        return f"Current Balance: {self.account_balance}"

    def view_transactions(self):
        """View transaction history"""
        return "\n".join([f"Transaction {i+1}: {amt}" for i, amt in enumerate(self.transactions)])

# Bot initialization and running
bot_token = "**********"
atm_bot = "**********"
atm_bot.bot.polling()