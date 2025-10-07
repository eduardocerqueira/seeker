#date: 2025-10-07T17:12:42Z
#url: https://api.github.com/gists/436858be4951cf26b542f364c3b9c1ce
#owner: https://api.github.com/users/shreyaghosh2024-crypto

import tkinter as tk
from tkinter import messagebox
import random

# ---------------- GAME DATA ----------------
questions = [
    {"clue": "An agreement among firms to fix prices or output.", "answer": "Collusion"},
    {"clue": "A group of firms working together to limit competition.", "answer": "Cartel"},
    {"clue": "Firms acting independently but following each other's prices.", "answer": "Oligopoly"},
    {"clue": "A market with a single seller and high barriers to entry.", "answer": "Monopoly"},
    {"clue": "Many firms selling identical products.", "answer": "Perfect Competition"},
    {"clue": "Firms selling similar but differentiated products.", "answer": "Monopolistic Competition"},
    {"clue": "Illegal cooperation to increase profits.", "answer": "Cartel"},
    {"clue": "When a firm sets prices in response to competitors.", "answer": "Price Leadership"},
    {"clue": "Collusion can lead to _____ for consumers.", "answer": "Higher Prices"},
    {"clue": "An organization of firms reducing competition among themselves.", "answer": "Cartel"},
]

options_pool = ["Perfect Competition", "Monopolistic Competition", "Oligopoly", "Monopoly", 
                "Cartel", "Collusion", "Price Leadership", "Higher Prices"]

titles = [
    (10, "Tycoon 💎"),
    (9, "Businessman 💼"),
    (7, "Trader 📈"),
    (5, "Learner 📚"),
    (0, "Beginner 😅")
]

# ---------------- VARIABLES ----------------
round_num = 0
score = 0
hearts = 0
timer = 20
timer_running = False
current_question = {}

# ---------------- FUNCTIONS ----------------
def start_game():
    global round_num, score, hearts
    round_num = 1
    score = 0
    hearts = 0
    intro_frame.pack_forget()
    game_frame.pack()
    next_round()

def next_round():
    global current_question, timer, timer_running
    if round_num > len(questions):
        end_game()
        return
    timer = 20
    timer_running = True
    current_question = questions[round_num - 1]
    clue_label.config(text=f"💡 Clue: {current_question['clue']}")
    # Prepare options (include correct + 3 random)
    options = [current_question["answer"]]
    while len(options) < 4:
        opt = random.choice(options_pool)
        if opt not in options:
            options.append(opt)
    random.shuffle(options)
    for i, btn in enumerate(answer_buttons):
        btn.config(text=options[i], bg="#F48FB1", state="normal")
    update_hearts()
    update_timer_label()
    root.after(1000, countdown)

def countdown():
    global timer, timer_running
    if timer_running:
        timer -= 1
        update_timer_label()
        if timer <= 0:
            timer_running = False
            show_correct_answer()
        else:
            root.after(1000, countdown)

def update_timer_label():
    timer_label.config(text=f"⏰ Time Left: {timer}")

def update_hearts():
    hearts_label.config(text=f"Hearts: {'❤️'*hearts}")

def check_answer(btn):
    global score, hearts, round_num, timer_running
    timer_running = False
    if btn.cget("text") == current_question["answer"]:
        btn.config(bg="#4CAF50")  # green correct
        score += 1
        hearts += 1
    else:
        btn.config(bg="#F44336")  # red wrong
        show_correct_answer()
    round_num += 1
    root.after(1000, next_round)

def show_correct_answer():
    for btn in answer_buttons:
        if btn.cget("text") == current_question["answer"]:
            btn.config(bg="#4CAF50")
        btn.config(state="disabled")
    root.after(1000, lambda: None)

def end_game():
    game_frame.pack_forget()
    title = "Beginner 😅"
    for t_score, t_name in titles:
        if score >= t_score:
            title = t_name
            break
    scoreboard = "\n".join([f"{i+1}. {q['clue']} → {q['answer']}" for i,q in enumerate(questions)])
    messagebox.showinfo("🎉 Game Over 🎉", f"Final Score: {score}/{len(questions)}\nTitle: {title}\n\nScoreboard:\n{scoreboard}")

# ---------------- GUI ----------------
root = tk.Tk()
root.title("🍬 Market Structure Mystery 🍬")
root.geometry("900x650")
root.configure(bg="#FCE4EC")  # pastel pink

# Intro Frame
intro_frame = tk.Frame(root, bg="#FCE4EC")
intro_frame.pack(fill="both", expand=True)
intro_label = tk.Label(intro_frame, text="🍬 Welcome to Market Structure Mystery 🍬\n\nRules:\n1. 10 rounds total.\n2. Read the clue and choose the correct answer.\n3. Timer: 20 seconds per round.\n4. Hearts indicate correct answers.\nGood Luck!", font=("Helvetica", 18), justify="center", bg="#FCE4EC")
intro_label.pack(pady=80)
start_btn = tk.Button(intro_frame, text="Start Game", font=("Helvetica", 18, "bold"), bg="#EC407A", fg="white", width=20, command=start_game)
start_btn.pack(pady=50)

# Game Frame
game_frame = tk.Frame(root, bg="#FCE4EC")
clue_label = tk.Label(game_frame, text="", font=("Helvetica", 20), bg="#FCE4EC")
clue_label.pack(pady=40)

answer_buttons = []
for i in range(4):
    btn = tk.Button(game_frame, text="", font=("Helvetica", 16, "bold"), bg="#F48FB1", fg="white", width=30, command=lambda b=i: check_answer(answer_buttons[b]))
    btn.pack(pady=10)
    answer_buttons.append(btn)

timer_label = tk.Label(game_frame, text=f"⏰ Time Left: {timer}", font=("Helvetica", 18), bg="#FCE4EC")
timer_label.pack(pady=20)

hearts_label = tk.Label(game_frame, text=f"Hearts: {'❤️'*hearts}", font=("Helvetica", 18), bg="#FCE4EC")
hearts_label.pack(pady=10)

# ---------------- START GUI ----------------
root.mainloop()
