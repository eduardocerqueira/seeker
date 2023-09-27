#date: 2023-09-27T17:01:11Z
#url: https://api.github.com/gists/f45cd57f77c5e2fd181cc847a2cb496c
#owner: https://api.github.com/users/Phil-Miles

from question_model import Question
from data import question_data
from quiz_brain import QuizBrain
from ui import QuizInterface

question_bank = []
for question in question_data:
    question_text = question["question"]
    question_answer = question["correct_answer"]
    new_question = Question(question_text, question_answer)
    question_bank.append(new_question)


quiz = QuizBrain(question_bank)
quiz_ui = QuizInterface(quiz)
