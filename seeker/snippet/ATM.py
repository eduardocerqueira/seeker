#date: 2024-06-18T17:01:47Z
#url: https://api.github.com/gists/ae02573e41281a263c86995f262d9edc
#owner: https://api.github.com/users/jaikishore3130

from datetime import datetime,date

class ATM:
    def __init__(self, account_number, pin, account_type, balance, transactions, last_transaction_date):
        self.account_number = account_number
        self.pin = pin
        self.account_type = account_type
        self.balance = balance
        self.transactions = transactions
        self.last_transaction_date = last_transaction_date
        
    def check_pin(self, input_pin):
        return self.pin == input_pin

    def check_balance(self, lang):
        translations = {
            'en': f"Your balance is: {self.balance}",
            'gu': f"તમારું બેલેન્સ {self.balance} છે",
            'hi': f"आपका बैलेंस है: {self.balance}",
            'kn': f"ನಿಮ್ಮ ಬ್ಯಾಲೆನ್ಸ್ ಆಗಿದೆ: {self.balance}",
            'ml': f"നിങ്ങളുടെ ബാലൻസ് ആണ്: {self.balance}",
            'mr': f"तुमची शिल्लक आहे: {self.balance}",
            'od': f"ତୁମର ସନ୍ତୁଳନ ହେଉଛି: {self.balance}",
            'ta': f"உங்கள் இருப்பு உள்ளது: {self.balance}",
            'te': f"మీ బ్యాలెన్స్: {self.balance}",
            'ur': f"آپ کا توازن: {self.balance}"
        }
        return translations.get(lang, 'en')

    def deposit(self, lang, amount):
        if  self.account_type=="saving":
            if amount > 100000:
                translations = {
                    'en': "You cannot deposit more than 100,000.",
                    'gu': "તમે 100,000 થી વધુ જમા કરી શકતા નથી.",
                    'hi': "आप 100,000 से अधिक जमा नहीं कर सकते.",
                    'kn': "ನೀವು 100,000 ಕ್ಕಿಂತ ಹೆಚ್ಚು ಠೇವಣಿ ಮಾಡಬಹುದು.",
                    'ml': "നിങ്ങൾക്ക് 100,000 ൽ കൂടുതൽ നിക്ഷേപിക്കാനാവില്ല.",
                    'mr': "तुम्ही 100,000 पेक्षा अधिक ठेवू शकत नाही.",
                    'od': "ଆପଣ 100,000 ରୁ ଅଧିକ ଦିନ୍ଦା ନାହିଁ |",
                    'ta': "நீங்கள் 100,000 க்கும் அதிகம் டெபாசிட் செய்ய முடியாது.",
                    'te': "మీరు 100,000 కంటే ఎక్కువ డిపాజిట్ చేయలేరు.",
                    'ur': "آپ 100,000 سے زیادہ جمع نہیں کر سکتے۔"
                }
                return translations.get(lang, 'en')
            else:
                self.balance += amount
                self.update_account()
                translations = {
                    'en': f"Deposit successful. Your new balance is: {self.balance}",
                    'gu': f"ડિપોઝિટ સફળ. તમારું નવું બેલેન્સ છે: {self.balance}",
                    'hi': f"जमा सफल. आपका नया बैलेंस है: {self.balance}",
                    'kn': f"ಠೇವಣಿ ಯಶಸ್ವಿಯಾಗಿದೆ. ನಿಮ್ಮ ಹೊಸ ಬ್ಯಾಲೆನ್ಸ್ ಆಗಿದೆ: {self.balance}",
                    'ml': f"നിക്ഷേപം വിജയിച്ചു. നിങ്ങളുടെ പുതിയ ബാലൻസ്: {self.balance}",
                    'mr': f"ठेव यशस्वी. तुमची नवीन शिल्लक आहे: {self.balance}",
                    'od': f"ସଫଳତାର ସହିତ ତୁମର ସନ୍ତୁଳନ ଜମା କର: {self.balance}",
                    'ta': f"டெபாசிட் வெற்றி. உங்கள் புதிய இருப்பு: {self.balance}",
                    'te': f"డిపాజిట్ విజయవంతమైంది. మీ కొత్త బ్యాలెన్స్: {self.balance}",
                    'ur': f"ڈپازٹ ہمارے بیلنس میں کامیاب: {self.balance}"
                }
                return translations.get(lang, 'en')
        else:
            if amount > 400000:
                translations = {
                    'en': "You cannot deposit more than 400,000.",
                    'gu': "તમે 400,000 થી વધુ જમા કરી શકતા નથી.",
                    'hi': "आप 400,000 से अधिक जमा नहीं कर सकते.",
                    'kn': "ನೀವು 400,000 ಕ್ಕಿಂತ ಹೆಚ್ಚು ಠೇವಣಿ ಮಾಡಬಹುದು.",
                    'ml': "നിങ്ങൾക്ക് 400,000 ൽ കൂടുതൽ നിക്ഷേപിക്കാനാവില്ല.",
                    'mr': "तुम्ही 400,000 पेक्षा अधिक ठेवू शकत नाही.",
                    'od': "ଆପଣ 400,000 ରୁ ଅଧିକ ଦିନ୍ଦା ନାହିଁ |",
                    'ta': "நீங்கள் 400,000 க்கும் அதிகம் டெபாசிட் செய்ய முடியாது.",
                    'te': "మీరు 100,000 కంటే ఎక్కువ డిపాజిట్ చేయలేరు.",
                    'ur': "آپ 400,000 سے زیادہ جمع نہیں کر سکتے۔"
                }
                return translations.get(lang, 'en')
            else:
                self.balance += amount
                self.update_account()
                translations = {
                    'en': f"Deposit successful. Your new balance is: {self.balance}",
                    'gu': f"ડિપોઝિટ સફળ. તમારું નવું બેલેન્સ છે: {self.balance}",
                    'hi': f"जमा सफल. आपका नया बैलेंस है: {self.balance}",
                    'kn': f"ಠೇವಣಿ ಯಶಸ್ವಿಯಾಗಿದೆ. ನಿಮ್ಮ ಹೊಸ ಬ್ಯಾಲೆನ್ಸ್ ಆಗಿದೆ: {self.balance}",
                    'ml': f"നിക്ഷേപം വിജയിച്ചു. നിങ്ങളുടെ പുതിയ ബാലൻസ്: {self.balance}",
                    'mr': f"ठेव यशस्वी. तुमची नवीन शिल्लक आहे: {self.balance}",
                    'od': f"ସଫଳତାର ସହିତ ତୁମର ସନ୍ତୁଳନ ଜମା କର: {self.balance}",
                    'ta': f"டெபாசிட் வெற்றி. உங்கள் புதிய இருப்பு: {self.balance}",
                    'te': f"డిపాజిట్ విజయవంతమైంది. మీ కొత్త బ్యాలెన్స్: {self.balance}",
                    'ur': f"ڈپازٹ ہمارے بیلنس میں کامیاب: {self.balance}"
                }
                return translations.get(lang, 'en')
            
    
    def withdraw(self, lang, amount):
        self.reset_transaction()
        current_date = date.today().strftime('%Y-%m-%d')
        if  self.account_type=="saving":
            
            if amount > 40000 or (self.last_transaction_date == current_date and self.transactions >= 10):
                
                translations = {
                    'en': "You cannot withdraw more than 40,000 in a day or exceed 10 transactions.",
                    'gu': "તમે એક દિવસમાં 40,000 થી વધુ ઉપાડી શકતા નથી અથવા 10 ટ્રાન્ઝેક્શન કરતાં વધુ કરી શકતા નથી.",
                    'hi': "आप एक दिन में 40,000 से अधिक नहीं निकाल सकते या 10 लेनदेन से अधिक नहीं कर सकते।",
                    'kn': "ನೀವು ಒಂದು ದಿನದಲ್ಲಿ 40,000 ಕ್ಕಿಂತ ಹೆಚ್ಚು ಹಿಂಪಡೆಯಲು ಅಥವಾ 10 ವಹಿವಾಟುಗಳನ್ನು ಮೀರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
                    'ml': "നിങ്ങൾക്ക് ഒരു ദിവസം 40,000 ൽ കൂടുതൽ പിൻവലിക്കാനോ 10 ഇടപാടുകൾക്ക് മുകളിലോ പോകാനാകില്ല.",
                    'mr': "तुम्ही एका दिवशी 40,000 पेक्षा अधिक पैसे काढू शकत नाही किंवा 10 व्यवहारांपेक्षा अधिक करू शकत नाही.",
                    'od': "ଆପଣ ଏକ ଦିନରେ 40,000 ରୁ ଅଧିକ ପ୍ରତ୍ୟାହାର କରିପାରିବେ ନାହିଁ | ଆପଣ 10 ଟି ଟ୍ରାନଜାକସନ ପାରି କରିପାରିବେ ନାହିଁ |",
                    'ta': "நீங்கள் ஒரு நாளில் 40,000 க்கும் மேற்பட்டதைத் திரும்பப் பெற முடியாது அல்லது 10 பரிவர்த்தனைகளை மீற முடியாது.",
                    'te': "మీరు ఒక రోజులో 40,000 కంటే ఎక్కువ ఉపసంహరించుకోలేరు లేదా 10 లావాదేవీలను మించకూడదు.",
                    'ur': "آپ ایک دن میں 40,000 سے زیادہ نہیں نکال سکتے یا 10 لین دین سے زیادہ نہیں کر سکتے۔"
                }
                
                return translations.get(lang, 'en')
            elif amount > self.balance:
                translations = {
                    'en': "Insufficient funds.",
                    'gu': "અપૂરતું ભંડોળ.",
                    'hi': "अपर्याप्त कोष।",
                    'kn': "ಅಪರಿಹಾರ್ಯ ಹಣಗಳು.",
                    'ml': "അപര്യാപ്തമായ ഫണ്ട്.",
                    'mr': "अपुरा निधी.",
                    'od': "ପର୍ଯ୍ୟାପ୍ତ ଅର୍ଥ",
                    'ta': "போதுமான பணம் இல்லை.",
                    'te': "తగినంత నిధులు లేవు.",
                    'ur': "ناکافی فنڈز۔"
                }
                return translations.get(lang, 'en')
            else:
                self.balance -= amount
                self.transactions += 1
                self.update_account()
                translations = {
                    'en': f"Withdrawal successful. Your new balance is: {self.balance}",
                    'gu': f"ઉપાડ સફળ. તમારું નવું બેલેન્સ છે: {self.balance}",
                    'hi': f"निकासी सफल। आपका नया बैलेंस है: {self.balance}",
                    'kn': f"ಹಿಂಪಡೆಯುವಿಕೆ ಯಶಸ್ವಿಯಾಗಿದೆ. ನಿಮ್ಮ ಹೊಸ ಬ್ಯಾಲೆನ್ಸ್: {self.balance}",
                    'ml': f"പിൻവലിക്കൽ വിജയിച്ചു. നിങ്ങളുടെ പുതിയ ബാലൻസ്: {self.balance}",
                    'mr': f"माघार यशस्वी. तुमची नवीन शिल्लक आहे: {self.balance}",
                    'od': f"ପ୍ରତ୍ୟାହାର ସଫଳତାର ସହିତ ତୁମର ସନ୍ତୁଳନ ଜମା କର: {self.balance}",
                    'ta': f"பணத்தை திரும்பப் பெறுதல் வெற்றிகரமாக முடிந்தது. உங்கள் புதிய இருப்பு: {self.balance}",
                    'te': f"విత్‌ డ్రా విజయవంతం. మీ కొత్త బ్యాలెన్స్: {self.balance}",
                    'ur': f"واپسی کامیاب ہوئی۔ آپ کا نیا توازن ہے: {self.balance}"
                }
                return translations.get(lang, 'en')
        else:
            if amount > 90000 or (self.last_transaction_date == current_date and self.transactions >= 10):
                
                translations = {
                    'en': "You cannot withdraw more than 90,000 in a day or exceed 10 transactions.",
                    'gu': "તમે એક દિવસમાં 90,000 થી વધુ ઉપાડી શકતા નથી અથવા 10 ટ્રાન્ઝેક્શન કરતાં વધુ કરી શકતા નથી.",
                    'hi': "आप एक दिन में 90,000से अधिक नहीं निकाल सकते या 10 लेनदेन से अधिक नहीं कर सकते।",
                    'kn': "ನೀವು ಒಂದು ದಿನದಲ್ಲಿ90,000 ಕ್ಕಿಂತ ಹೆಚ್ಚು ಹಿಂಪಡೆಯಲು ಅಥವಾ 10 ವಹಿವಾಟುಗಳನ್ನು ಮೀರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
                    'ml': "നിങ്ങൾക്ക് ഒരു ദിവസം 90,000 ൽ കൂടുതൽ പിൻവലിക്കാനോ 10 ഇടപാടുകൾക്ക് മുകളിലോ പോകാനാകില്ല.",
                    'mr': "तुम्ही एका दिवशी 90,000 पेक्षा अधिक पैसे काढू शकत नाही किंवा 10 व्यवहारांपेक्षा अधिक करू शकत नाही.",
                    'od': "ଆପଣ ଏକ ଦିନରେ 90,000 ରୁ ଅଧିକ ପ୍ରତ୍ୟାହାର କରିପାରିବେ ନାହିଁ | ଆପଣ 10 ଟି ଟ୍ରାନଜାକସନ ପାରି କରିପାରିବେ ନାହିଁ |",
                    'ta': "நீங்கள் ஒரு நாளில் 90,000 க்கும் மேற்பட்டதைத் திரும்பப் பெற முடியாது அல்லது 10 பரிவர்த்தனைகளை மீற முடியாது.",
                    'te': "మీరు ఒక రోజులో 90,000కంటే ఎక్కువ ఉపసంహరించుకోలేరు లేదా 10 లావాదేవీలను మించకూడదు.",
                    'ur': "آپ ایک دن میں 90,000 سے زیادہ نہیں نکال سکتے یا 10 لین دین سے زیادہ نہیں کر سکتے۔"
                }
                return translations.get(lang, 'en')
            elif amount > self.balance:
                translations = {
                    'en': "Insufficient funds.",
                    'gu': "અપૂરતું ભંડોળ.",
                    'hi': "अपर्याप्त कोष।",
                    'kn': "ಅಪರಿಹಾರ್ಯ ಹಣಗಳು.",
                    'ml': "അപര്യാപ്തമായ ഫണ്ട്.",
                    'mr': "अपुरा निधी.",
                    'od': "ପର୍ଯ୍ୟାପ୍ତ ଅର୍ଥ",
                    'ta': "போதுமான பணம் இல்லை.",
                    'te': "తగినంత నిధులు లేవు.",
                    'ur': "ناکافی فنڈز۔"
                }
                return translations.get(lang, 'en')
            else:
                self.balance -= amount
                self.transactions += 1
                self.update_account()
                translations = {
                    'en': f"Withdrawal successful. Your new balance is: {self.balance}",
                    'gu': f"ઉપાડ સફળ. તમારું નવું બેલેન્સ છે: {self.balance}",
                    'hi': f"निकासी सफल। आपका नया बैलेंस है: {self.balance}",
                    'kn': f"ಹಿಂಪಡೆಯುವಿಕೆ ಯಶಸ್ವಿಯಾಗಿದೆ. ನಿಮ್ಮ ಹೊಸ ಬ್ಯಾಲೆನ್ಸ್: {self.balance}",
                    'ml': f"പിൻവലിക്കൽ വിജയിച്ചു. നിങ്ങളുടെ പുതിയ ബാലൻസ്: {self.balance}",
                    'mr': f"माघार यशस्वी. तुमची नवीन शिल्लक आहे: {self.balance}",
                    'od': f"ପ୍ରତ୍ୟାହାର ସଫଳତାର ସହିତ ତୁମର ସନ୍ତୁଳନ ଜମା କର: {self.balance}",
                    'ta': f"பணத்தை திரும்பப் பெறுதல் வெற்றிகரமாக முடிந்தது. உங்கள் புதிய இருப்பு: {self.balance}",
                    'te': f"విత్‌ డ్రా విజయవంతం. మీ కొత్త బ్యాలెన్స్: {self.balance}",
                    'ur': f"واپسی کامیاب ہوئی۔ آپ کا نیا توازن ہے: {self.balance}"
                }
                return translations.get(lang, 'en')
            
    def reset_transaction(self):
        current_date = date.today().strftime('%Y-%m-%d')
        if self.last_transaction_date != current_date:
            self.transactions = 0
            self.update_account()
            
    
    def update_account(self):
        with open('accounts.txt', 'r') as file:
            lines = file.readlines()
        with open('accounts.txt', 'w') as file:
            for line in lines:
                account_data = line.strip().split(',')
                if account_data[0] == self.account_number:
                    account_data[3] = str(self.balance)
                    account_data[4] = str(self.transactions)
                    account_data[5] = date.today().strftime('%Y-%m-%d')
                    line = ','.join(account_data) + '\n'
                file.write(line)
    
    @staticmethod
    def load_account(account_number):
        with open('accounts.txt', 'r') as file:
            for line in file:
                account_data = line.strip().split(',')
                if account_data[0] == account_number:
                    return ATM(
                        account_number=account_data[0],
                        pin=account_data[1],
                        account_type=account_data[2],
                        balance=float(account_data[3]),


                        transactions=int(account_data[4]),
                        last_transaction_date=account_data[5]
                    )
        return None

def language():                             #function for selecting language
        print("\nSelect your language:")
        print("1. English")
        print("2. ગુજરાતી(Gujarati)")
        print("3.हिंदी(Hindi)")
        print("4. ಕನ್ನಡ (Kannada)")
        print("5. മലയാളം(Malayalam)")
        print("6. मराठी (Marathi)")
        print("7.ଓଡ଼ିଆ(odia)")
        print("8. தமிழ்(Tamil)")
        print("9.తెలుగు(Telugu)")
        print("10.اُردُو(urdu)")
        choice=int(input("Enter the number corresponding to your language choice:"))
        print("\n")
        language={1:'en',2:'gu',3:'hi',4:'kn',5:'ml',6:'mr',7:'od',8:'ta',9:'te',10:'ur'}
        return language.get(choice,'en')

def  account(lang):
    translations={
        'en':"Select your account type:",
        'gu':"તમારા એકાઉન્ટનો પ્રકાર પસંદ કરો:",
        'hi':"अपना खाता प्रकार चुनें:",
        'kn':"ನಿಮ್ಮ ಖಾತೆಯ ಪ್ರಕಾರವನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
        'ml':"നിങ്ങളുടെ അക്കൗണ്ട് തരം തിരഞ്ഞെടുക്കുക: ",
        'mr':"तुमचा खाते प्रकार निवडा : ",
        'od':"ଆପଣଙ୍କର ଖାତା Tepe ଚୟନ କରନ୍ତୁ |",
        'ta':"உங்கள் கணக்கு வகையைத் தேர்ந்தெடுக்கவும்:",
        'te':"మీ ఖాతా రకాన్ని ఎంచుకోండి:",
        'ur':"  اپنے اکاؤنٹ کی قسم منتخب کریں۔"
        }
    print(translations.get(lang,'en'))
    trans={
        'en':["1. Saving Account", "2. Current Account"],
        'gu':["1. બચત ખાતું", "2. ચાલુ ખાતું"],
        'hi':["1. बचत खाता", "2. चालू खाता"],
        'kn':["1. ರಕ್ಷಣಾತ್ಮಕ ಖಾತೆ", "2. ಪ್ರಸ್ತುತ ಖಾತೆ"],
        'ml':["1. സേവിംഗ് അക്കൗണ്ട്", "2. കറൻ്റ് അക്കൗണ്ട്"],
        'mr':["1. बचत खाते", "2. चालू खाते"],
        'od':["1. ଗୀତ ଖାତା |", "2. ସାମ୍ପ୍ରତିକ ଆକାଉଣ୍ଟ୍ |"],
        'ta':["1. சேமிப்பு கணக்கு", "2. நடப்புக் கணக்கு"],
        'te':["1.పొదుపు ఖాతా ", "2. వాడకం ఖాతా"],
        'ur':["1. بچت اکاؤنٹ", "2. موجودہ اکاؤنٹ"]
        }
    for i in trans.get(lang):
          print(i)
    transop={
        'en':"Enter your choice(1/2):",
        'gu':"તમારી પસંદગી દાખલ કરો (1/2):",
        'hi':"अपनी पसंद दर्ज करें (1/2):",
        'kn':" ನಿಮ್ಮ ಆಯ್ಕೆಯನ್ನು ನಮೂದಿಸಿ(1/2):",
        'ml':"നിങ്ങളുടെ ചോയ്സ് നൽകുക (1/2): ",
        'mr':"तुमची निवड प्रविष्ट करा (1/2):  ",
        'od':"ତୁମର ଚାରିଟି ପ୍ରବେଶ କର |(1/2):",
        'ta':" உங்கள் விருப்பத்தை உள்ளிடவும் (1/2):",
        'te':"మీ ఎంపికను నమోదు చేయండి (1/2):",
        'ur':" اپنی پسند درج کریں۔ (1/2):"
        }
    choice= int(input(transop.get(lang,'en')))
    while True:
        if choice==1:
            x="saving"
            break
        elif choice==2:
            x="current"
            break
        else:
            print("inavlid")
            choice= int(input(transop.get(lang,'en')))
        
    return x

def main():
    lang =language()
    atm = account_number = input("Enter Account Number: ")
    atm = ATM.load_account(account_number)
    account_type =account(lang)
    if atm and atm.account_type == account_type:
        print("Account verification successful")

        while True:
            print("\n**** Welcome to the ATM ****")
            translations = {
                'en':"1.Check Balance \n2.Deposit\n3.Withdraw\n4.Exit",
                'gu':"1.બેલેન્સ તપાસો\n2. જમા\n3. પાછી ખેંચો\n4. બહાર નીકળો(exit)",
                'hi':"1. बैलेंस चेक करें\n2. जमा\n3. वापस लेना\n4. बाहर निकलें(exit)",
                'kn':"1.ಬ್ಯಾಲೆನ್ಸ್ ಪರಿಶೀಲಿಸಿ \n2.ಠೇವಣಿ\n3.ಹಿಂತೆಗೆದುಕೊಳ್ಳಿ\n4.ನಿರ್ಗಮಿಸಿ (exit)", 
                'ml':"1. ബാലൻസ് പരിശോധിക്കുക\n2. നിക്ഷേപം \n3. പിൻവലിക്കുക \n4. പുറത്തുകടക്കുക(exit)",
                'mr':"1. शिल्लक तपासा\n2. ठेव\n3. मागे घ्या \n4. बाहेर पडा(Exit) ",
                'od':"1. ସନ୍ତୁଳନ ଯାଞ୍ଚ କରନ୍ତୁ |\n2. ଜମା\n3. ପ୍ରତ୍ୟାହାର\n4. ଚିଟ୍(exit)",
                'ta':"1.இருப்பை சரிபார்க்கவும்\n2.வைப்பு\n3.திரும்பப் பெறவும்\n4.வெளியேறு(exit)",
                'te':"1. బ్యాలెన్స్ తనిఖీ చేయండి\n2. డిపాజిట్\n3. ఉపసంహరించుకోండి\n4. నిష్క్రమించు(exit)",
                'ur':" ا1. بیلنس چیک کریں\n2. جمع\n3. واپس لے لو\n4. باہر نکلیں(exit)"
                }
            print(translations.get(lang, 'en'))
            trans={
                'en':"Enter your choice(1/2/3/4):",
                'gu':"તમારી પસંદગી દાખલ કરો (1/2/3/4):",
                'hi':"अपनी पसंद दर्ज करें (1/2/3/4):",
                'kn':" ನಿಮ್ಮ ಆಯ್ಕೆಯನ್ನು ನಮೂದಿಸಿ(1/2/3/4):",
                'ml':"നിങ്ങളുടെ ചോയ്സ് നൽകുക (1/2/3/4): ",
                'mr':"तुमची निवड प्रविष्ट करा (1/2/3/4):  ",
                'od':"ତୁମର ଚାରିଟି ପ୍ରବେଶ କର |(1/2/3/4):",
                'ta':" உங்கள் விருப்பத்தை உள்ளிடவும் (1/2/3/4):",
                'te':"మీ ఎంపికను నమోదు చేయండి (1/2/3/4):",
                'ur':" اپنی پسند درج کریں۔ (1/2/3/4):"
                }

            choice = int(input(trans.get(lang,'en')))
            match(choice):
                case 1:
                     pin = input("Enter PIN: ")
                     if atm.check_pin(pin):
                         print(atm.check_balance(lang))
                     else:
                         print("Incorrect PIN")
                case 2:
                    translations={
                        'en':"Enter amount to deposit:",
                        'gu':"જમા કરવા માટે રકમ દાખલ કરો:",
                        'hi':"जमा करने के लिए राशि दर्ज करें:",
                        'kn':"ಠೇವಣಿ ಮಾಡಲು ಮೊತ್ತವನ್ನು ನಮೂದಿಸಿ: ",
                        'ml':"നിക്ഷേപിക്കാനുള്ള തുക നൽകുക: ",
                        'mr':"जमा करण्यासाठी रक्कम प्रविष्ट करा: ",
                        'od':"ଜମା ପାଇଁ ଇଣ୍ଟର ରାଶି |",
                        'ta':"டெபாசிட் செய்ய வேண்டிய தொகையை உள்ளிடவும்",
                        'te':"డిపాజిట్ చేయడానికి మొత్తాన్ని నమోదు చేయండి:",
                        'ur':" جمع کرنے کے لیے رقم درج کریں۔"
                        }
                    amount=float(input(translations.get(lang,'en')))
                    pin = input("Enter PIN: ")
                    if atm.check_pin(pin):
                        print(atm.deposit(lang,amount))
                    else:
                        print("Incorrect PIN ")
                case 3:
                    translations={
                        'en':"Enter amount to withdraw:",
                        'gu':"ઉપાડવા માટે રકમ દાખલ કરો:",
                        'hi':"निकालने के लिए राशि दर्ज करें :",
                        'kn':"ಹಿಂಪಡೆಯಲು ಮೊತ್ತವನ್ನು ನಮೂದಿಸಿ: ",
                        'ml':"പിൻവലിക്കാനുള്ള തുക നൽകുക:  ",
                        'mr':"काढण्यासाठी रक्कम प्रविष्ट करा : ",
                        'od':"ୱିଥ୍ରୋ ସହିତ ଇଣ୍ଟର ପରିମାଣ |",
                        'ta':" திரும்பப் பெறுவதற்கான தொகையை உள்ளிடவும்",
                        'te':"ఉపసంహరించుకోవడానికి మొత్తాన్ని నమోదు చేయండి:",
                        'ur':"  نکالنے کے لیے رقم درج کریں۔"
                        }
                    amount=float(input(translations.get(lang,'en')))
                    pin = input("Enter PIN: ")
                    if atm.check_pin(pin):
                        print(atm.withdraw(lang,amount))
                    else:
                        print("Incorrect PIN")
                case 4:
                    thankyou={
                        'en':"Thank you for using the ATM. ",
                        'gu':"એટીએમ નો ઉપયોગ કરવા બદલ આભાર. ",
                        'hi':"एटीएम का उपयोग करने के लिए धन्यवाद. ",
                        'kn':"  ಎಟಿಎಂ ಬಳಸಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು.",
                        'ml':"എടിഎം ഉപയോഗിച്ചതിന് നന്ദി. ",
                        'mr':"ATM वापरल्याबद्दल धन्यवाद. ",
                        'od':"ଶ୍ରେଷ୍ଠ ହୋଇଥିବାରୁ ଧନ୍ୟବାଦ ",
                        'ta':"ஏடிஎம் பயன்படுத்தியதற்கு நன்றி.  ",
                        'te':"ఎటిఎం ని ఉపయోగించినందుకు ధన్యవాదాలు. ",
                        'ur':"  اے ٹی ایم الوداع استعمال کرنے کے لیے آپ کا شکریہ"
                        }
                    print(thankyou.get(lang,'en'))
                    exit()
                case _:
                    inv={
                        'en':"Invalid choice. Please enter a valid option.",
                        'gu':" અમાન્ય પસંદગી. કૃપા કરીને માન્ય વિકલ્પ દાખલ કરો. ",
                        'hi':"एटीएम का उपयोग करने के लिए धन्यवाद. ",
                        'kn':" ಅಮಾನ್ಯ ಆಯ್ಕೆ ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ ಆಯ್ಕೆಯನ್ನು ನಮೂದಿಸಿ",
                        'ml':"അസാധുവായ തിരഞ്ഞെടുപ്പ്. സാധുവായ ഒരു ഓപ്ഷൻ നൽകുക.",
                        'mr':"अवैध निवड. कृपया एक वैध पर्याय प्रविष्ट करा.",
                        'od':" ଦୟାକରି ପ୍ୟାରେଣ୍ଟ୍ ଅପ୍ସନ୍ ପ୍ରବେଶ କରନ୍ତୁ |",
                        'ta':" தவறான தேர்வு.  சரியான விருப்பத்தை உள்ளிடவும்.",
                        'te':"చెల్లని ఎంపిక. దయచేసి చెల్లుబాటు అయ్యే ఎంపికను నమోదు చేయండి. ",
                        'ur':" غلط انتخاب برائے مہربانی ایک درست آپشن درج کریں۔"
                        }
                        
                    print(inv.get(lang,'en'))
                
    else:
        print("Account verification failed")
       
      
if __name__ == "__main__":
    main()


