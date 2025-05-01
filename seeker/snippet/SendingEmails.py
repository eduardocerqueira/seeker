#date: 2025-05-01T17:00:36Z
#url: https://api.github.com/gists/48f38836ae2c8b1d7ed5a9ac881660da
#owner: https://api.github.com/users/Hasanat-Ashraf

# **********************************************
# Name: Hasanat Ashraf
# Date: 5/1/2025
# Course Number: CSC-114-D01
# Course Name: Intermediate Topics in Python
# Problem Number: 10
# Email: hashraf2101@student.stcc.edu
# sending emails
# **********************************************
# imports here

import tkinter as tk
from tkinter import font

# Define as many constants or variables you need here

TITLE = "Sending Emails, Phase 1"
CONTINUE_PROMPT = "Do this again? [y/N] "

# **********************************************
# Define as many functions or classes you need here

class MailApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(TITLE)
        self.root.geometry("620x440")
        self.root.config(bg="#E6F2FF")

        self.to_input = tk.StringVar(value="example@domain.com")
        self.from_input = tk.StringVar(value="me@domain.com")
        self.subject_input = tk.StringVar(value="Write your subject here")
        self.status_msg = tk.StringVar()

        self.build_gui()
        self.root.mainloop()

    def build_gui(self):
        header_font = font.Font(size=15, weight="bold")
        label_font = font.Font(size=11, weight="bold")

        tk.Label(self.root, text="Send a Message", bg="#E6F2FF", font=header_font).pack(pady=10)

        form_frame = tk.Frame(self.root, bg="#E6F2FF")
        form_frame.pack(pady=5)

        tk.Label(form_frame, text="To:", font=label_font, bg="#E6F2FF").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(form_frame, textvariable=self.to_input, width=50).grid(row=0, column=1, padx=5)

        tk.Label(form_frame, text="From:", font=label_font, bg="#E6F2FF").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(form_frame, textvariable=self.from_input, width=50).grid(row=1, column=1, padx=5)

        tk.Label(form_frame, text="Subject:", font=label_font, bg="#E6F2FF").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(form_frame, textvariable=self.subject_input, width=50).grid(row=2, column=1, padx=5)

        tk.Label(self.root, text="Message Body:", font=label_font, bg="#E6F2FF").pack(anchor="w", padx=25, pady=(10, 0))
        self.message_box = tk.Text(self.root, height=8, width=72, font=("Arial", 11))
        self.message_box.insert(tk.END, "Type your message here...")
        self.message_box.pack(padx=25, pady=5)

        send_btn = tk.Button(self.root, text="Send", bg="#006699", fg="white", font=label_font, command=self.fake_send)
        send_btn.pack(pady=12)

        tk.Label(self.root, text="Status:", bg="#E6F2FF", font=label_font).pack(anchor="w", padx=25, pady=(5, 0))
        tk.Entry(self.root, textvariable=self.status_msg, state='readonly', font=("Arial", 10), width=60).pack(padx=25)

    def fake_send(self):
        to = self.to_input.get()
        sender = self.from_input.get()
        subject = self.subject_input.get()
        body = self.message_box.get("1.0", tk.END).strip()

        self.message_box.delete("1.0", tk.END)
        preview = f"To: {to}\nFrom: {sender}\nSubject: {subject}\n\n{body}"
        self.message_box.insert(tk.END, preview)
        self.status_msg.set("Message prepared (not sent) – check content above.")

# **********************************************
# Start your logic coding in the process function
def process():
    MailApp()

# **********************************************
# Do not change the do_this_again function
def do_this_again(prompt):
    do_over = input(prompt)
    return do_over.strip().lower() == 'y'

# **********************************************
# Do not change the main function
def main():
    print(f"Welcome to {TITLE}")
    while True:
        process()
        if not do_this_again(CONTINUE_PROMPT):
            break
    print(f"Thank you for using {TITLE}")

if __name__ == "__main__":
    main()
