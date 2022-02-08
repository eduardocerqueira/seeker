#date: 2022-02-08T17:14:02Z
#url: https://api.github.com/gists/126dc9bd1f44bc0f53ab4576f0205709
#owner: https://api.github.com/users/jaskaranSM

from tkinter import Tk, W, Label, StringVar, RIGHT, Entry, E, Button #Importing neccessary tkinter components

class LoanCalculator:
    def __init__(self, window): #the class constructor takes in a tkinter.TK object
        self.window = window

        #storage structures for storing components, might come in handy later 
        self.labels = [] 
        self.entries = []
        self.result_btn = None

        #python strings are immuatable, however class objects are not, wrapping them with a class object to get around that.
        self.annualInterestRateVar = StringVar()
        self.yearsVar = StringVar()
        self.amountVar = StringVar()
        self.monthlyPaymentAmountVar = StringVar()
        self.totalPaymentVar = StringVar()

        #initializer methods being called here
        self.__init__window()
        self.__init__labels()
        self.__init__entries()
        self.__init__buttons()

    def __init__labels(self): #this method places all the required labels on the window
        self.labels.append(self.get_label("Annual Interest Rate: ", W))
        self.labels.append(self.get_label("Years: ", W, row=2))
        self.labels.append(self.get_label("Amount: ", W, row=3))
        self.labels.append(self.get_label("Monthly Payment Amount: ", W, row=4))
        self.labels.append(self.get_label("Total Payment: ", W, row=5))

    def __init__entries(self): #this method places all the required entries on the window
        self.entries.append(self.get_entry(self.annualInterestRateVar, row=1))
        self.entries.append(self.get_entry(self.yearsVar, row=2))
        self.entries.append(self.get_entry(self.amountVar, row=3))
        self.entries.append(self.get_entry(self.monthlyPaymentAmountVar, row=4))
        self.entries.append(self.get_entry(self.totalPaymentVar, row=5))
    
    def __init__window(self): #this method initializes the window with some default settings
        self.window.geometry("400x200")
        self.window.title("Loan Calculator - Created by Jaskaran")
 
    def __init__buttons(self): #this method places all the required buttons on the window
        self.result_btn = self.get_button("Compute Result", self.onButtonClickCallback)

    def onButtonClickCallback(self): #this is a callback method, whose function pointer is going to be 
                                        #passed to button command parameter.
        print("[onButtonClickCallback]: Clicked")

    def get_label(self, text, W, row = 1, column=1): #utility method to create label
        return Label(self.window, text = text).grid(row = row,
                                       column = column, sticky = W)

    def get_entry(self, tv, row=1, column=2, justify=RIGHT): #utility method to create entry
        return Entry(self.window, textvariable = tv,
            justify = justify).grid(row = row, column = column)

    def get_button(self, text, command, row=6, column=2, sticky=E): #utility method to create button
        return Button(self.window, text=text,
                           command = command).grid(
                               row = row, column = column, sticky = sticky)

    def mainloop(self): #this method calls the inner window's event loop method to start listening for events.
        self.window.mainloop()




sag = LoanCalculator(Tk())
sag.mainloop()