#date: 2022-11-24T17:04:41Z
#url: https://api.github.com/gists/806bdcf4fe245e6fe36105004f83ec28
#owner: https://api.github.com/users/QOULMPEFA

# Name: YU Jia le
# 2022/11/14 12:36:26
filename1 = 'PhoneBk.txt'
filename2 = "**********"


def login_main():
    while True:
        try:
            username_password = "**********"
        except:
            answer = "**********"
            if answer == 'y':
                username = input("Please input your username:")
                registration(username)
            else:
                print("Goodbye!")
                break
            username = input("Please input your username:")
            registration(username)
        username_list = []  # list to store the username
        password_list = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
            a, b = "**********"
            b = "**********"
            username_list.append(a)
            password_list.append(b)
            username_password_dict = "**********"
            # print(data)
        username = input("Please input your username:")
        if username in username_list:
            if login(username):
                main()
            else:
                break
        else:
            registration(username)


def registration(username):
    """A function to register a new user on the database and log them in afterwards."""
    username = username
    username_password = "**********"
    registration_pd = input('Please set your password: "**********"
    registration_pd_confirm = input('Please confirm your password: "**********"
    if registration_pd == registration_pd_confirm:
        print('Passwords confirmed')
        username_password.write(username + ", " + str(
            registration_pd) + '\n')
        # string function used to convert a possibe interger or floating point to a normal number
        # cannot do multiple arguements as it is not allowed
        username_password.close()
        print('Registration confirmed!')
        login(username)  # calls the log in function log them in the system.
    else:
        print('Passwords do not match!')
        registration(username)


def login(username):
    while True:
        """A function to operate the login feature, mainly asking for username and loging in once authenticated"""
        try:
            username_password = "**********"
        except:
            username_password = "**********"='utf-8')
        username_list = []  # list to store the username
        password_list = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"i "**********"  "**********"i "**********"n "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
            a, b = "**********"
            b = "**********"
            username_list.append(a)
            password_list.append(b)
            username_password_dict = "**********"
            # print(data)
        if username in username_list:
            for i in range(1, 4):
                password = input('The username is already in the database please enter password: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"= "**********"= "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"d "**********"i "**********"c "**********"t "**********"[ "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"] "**********": "**********"
                    print('\n')  # to make some space between output
                    print('Log in Successful!')
                    username_password.close()
                    return True
                else:
                    print("Account or password are not correct!")
                    if i < 3:
                        print('You have {0} chances'.format(3 - i))
            print('You have failed to provide us with a password in 3 tries!')
            return False
        else:
            print('Your username is not in the database!\n'
                  'Please register your username and password')
            username_password.close()  # closing the database
            registration(username)


def main():
    while True:
        menu()
        choice = input('\nPlease select:\n')
        if choice in ['1', '2', '3', '4', '5', '6', '7', '0']:
            if choice == '0':
                answer = input('Are you sure to exit phone organiser system？y/n\n')
                if answer == 'y':
                    print("Goodbye!")
                    break
                else:
                    continue
            elif choice == '1':
                phoneRec = []
                while True:
                    phoneRec.append(add())
                    answer = input("Do you want to add another phone record?y/n\n")
                    if answer == 'y':
                        continue
                    else:
                        break
                save(phoneRec)
                print("Phone record added!\n")
            elif choice == '2':
                removes()
            elif choice == '3':
                displays_time()
            elif choice == '4':
                checks()
            elif choice == '5':
                displays_nickname()
            elif choice == '6':
                copy()
            elif choice == '7':
                show_phoneRec()
        else:
            print('Wrong input, please try again!\n\n')
            pass


def menu():
    title = 'Phone Organiser'
    sub_title = 'Function Menu'
    print('{0}'.format(title).center(80, '=') + '\n')
    print('{0}'.format(sub_title).center(80, '-'))
    print('1. Add a new phone record')
    print('2. removes the phone record')
    print('3. displays a sorted list of phone records based on the last-call date and time')
    print('4. checks the validity of all input email addresses under a user given group')
    print('5. displays a sorted list of phone records based on the nickname')
    print('6. copies the phone record of a user-input phone number')
    print('7. display the phone records\' information under a user-given group')
    print('0. Exit\n')
    print('=' * 80)


def add():
    global a
    name = input('Please input name:\n')
    nickname = input('Please input nickname:\n')
    phone_number = input('Please input phone_number:\n')
    email = input('Please input email:\n')
    day = input('please input day:\n')
    month = input('please input month:\n')
    month = str(month)
    month_number = 0
    month_dict = dict(January=1, February=2, March=3, April=4, May=5, June=6, July=7, August=8, September=9,
                      October=10, November=11, December=12)
    while month_number != range(1, 13):
        if month == 'January' and 'February' and 'March' and 'April' and 'May' and 'June' and 'July' and 'August' \
                and 'September' and 'October' and 'November' and 'December':
            month_number = month_dict.get(month)
            break
        elif 1 <= int(month) <= 12:
            month_number = month
            break
        else:
            print("Wrong input, please try again!")
            month = input('please input month:\n')

    year = input('Please input year:\n')
    time = input('please input time:\n')
    date_time = "{0}{1}{2}{3}".format(year[2:4:1], day, month_number, time)
    print(date_time)
    print("_" * 80)
    group = input('''
Which group you want to add in?
a) Family
b) Friend
c) Junk
''')
    while True:
        if group == 'a':
            a = Family(name, nickname, phone_number, email, date_time)
            return a.phoneRec_Family()
        elif group == 'b':
            a = Friend(name, nickname, phone_number, email, date_time)
            return a.phoneRec_Friend()
        elif group == 'c':
            a = Junk(name, nickname, phone_number, email, date_time)
            return a.phoneRec_Junk()
        else:
            print('Wrong input, please try again!')
            group = input('''
Which group you want to add in?
a) Family
b) Friend
c) Junk
''')


def save(lst):
    try:
        PhoneBk = open(filename1, 'a', encoding='utf-8')
    except:
        PhoneBk = open(filename1, 'w', encoding='utf-8')
    for item in lst:
        PhoneBk.write(str(item) + '\n')
    PhoneBk.close()


def removes():
    pass


def displays_time():
    pass


def checks():
    pass


def displays_nickname():
    phoneRec = []
    try:
        phoneBK = open(filename1, 'r', encoding='utf-8')
        phoneBK_line = phoneBK.readlines()
        for item in phoneBK_line:
            phoneRec_dict = dict(eval(item))
            phoneRec.append(phoneRec_dict)
            phoneRec.sort(key=lambda x: x['nickname'], reverse=False)
            show(phoneRec)
    except:
        print("\'phoneBk\' not found!")


def copy():
    pass


def show_phoneRec():
    Family_list = []
    Friend_list = []
    Junk_list = []
    try:
        phoneBK = open(filename1, 'r', encoding='utf-8')
        phoneBK_line = phoneBK.readlines()
        group = input('''
        Which group you want to check?
        a) Family
        b) Friend
        c) Junk
        d) All of them
        ''')
        if group == 'a':
            for item in phoneBK_line:
                phoneBK_dict = dict(eval(item))
                if phoneBK_dict['group'] == 'Family':
                    Family_list.append(phoneBK_dict)
            print(Friend_list)
            show(Family_list)
        if group == 'b':
            for item in phoneBK_line:
                phoneBK_dict = dict(eval(item))
                if phoneBK_dict['group'] == 'Friend':
                    Friend_list.append(phoneBK_dict)
            print(Friend_list)
            show(Friend_list)
        if group == 'c':
            for item in phoneBK_line:
                phoneBK_dict = dict(eval(item))
                if phoneBK_dict['group'] == 'Junk':
                    Junk_list.append(phoneBK_dict)
            print(Junk_list)
            show(Junk_list)
        if group == 'd':
            for item in phoneBK_line:
                phoneBK_dict = dict(eval(item))
                if phoneBK_dict['group'] == 'Family':
                    Family_list.append(phoneBK_dict)
            print('Family'.center(60, '='))
            print(Family_list)
            show(Family_list)
            for item in phoneBK_line:
                phoneBK_dict = dict(eval(item))
                if phoneBK_dict['group'] == 'Junk':
                    Junk_list.append(phoneBK_dict)
            print('Junk'.center(60, '='))
            print(Junk_list)
            show(Junk_list)
            for item in phoneBK_line:
                phoneBK_dict = dict(eval(item))
                if phoneBK_dict['group'] == 'Junk':
                    Junk_list.append(phoneBK_dict)
            print('Junk'.center(60, '='))
            print(Junk_list)
            show(Junk_list)
    except:
        print("/'phoneBk/' not found!")


def show(Group_list):
    if len(Group_list) == 0:
        print('No phone record founded!')
        return
    print('{:^16}\t{:^16}'.format('nickname', 'phone_number'))
    print("-" * 36)
    for item in Group_list:
        print('{:^16}\t{:^16}'.format(item.get('nickname'), item.get('phone_number')))


class phoneBk:
    def __init__(self, name, nickname, phone_number, email, date_time):
        self.name = name
        self.nickname = nickname
        self.phone_number = phone_number
        self.email = email
        self.date_time = date_time


class Family(phoneBk):
    def __init__(self, name, nickname, phone_number, email, date_time):
        super().__init__(name, nickname, phone_number, email, date_time)

    def phoneRec_Family(self):
        return {'group': 'Family', 'name': self.name, 'nickname': self.nickname, 'phone_number': self.phone_number,
                'mail': self.email, 'date_time': self.date_time}


class Friend(phoneBk):
    def __init__(self, name, nickname, phone_number, email, date_time):
        super().__init__(name, nickname, phone_number, email, date_time)

    def phoneRec_Friend(self):
        return {'group': 'Friend', 'name': self.name, 'nickname': self.nickname, 'phone_number': self.phone_number,
                'mail': self.email, 'date_time': self.date_time}


class Junk(phoneBk):
    def __init__(self, name, nickname, phone_number, email, date_time):
        super().__init__(name, nickname, phone_number, email, date_time)

    def phoneRec_Junk(self):
        return {'group': 'Junk', 'name': self.name, 'nickname': self.nickname, 'phone_number': self.phone_number,
                'mail': self.email, 'date_time': self.date_time}


login_main()

