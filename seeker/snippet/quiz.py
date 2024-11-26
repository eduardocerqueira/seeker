#date: 2024-11-26T17:04:58Z
#url: https://api.github.com/gists/323e33972269504647fae074db8b2037
#owner: https://api.github.com/users/jeromee-dev

class Quiz:
  def __init__(self, question_list):
    self.question_list = question_list
    self.highscore = 0
    
  def ask_questions(self):
    current_score = 0
    for question in self.question_list:
      print(question.question)
      response = input()
      if question.check_answer_string(response):
        current_score += 1
    self.set_highscore(current_score)
    score_percentage = current_score / len(self.question_list) * 100
    print(f'You scored {current_score} out of {len(self.question_list)} ({score_percentage}%)')
  
  def reset_highscore(self):
    self.highscore = 0
    
  def set_highscore(self, score):
    if score > self.highscore:
      highscore = score
  

class Question:
  def __init__(self, question, answer_list, correct_answer):
    self.question = question
    self.answer_list = answer_list
    self.correct_answer = correct_answer
    
  def check_answer_string(self, user_answer):
    return user_answer == self.correct_answer
  
  def check_answer_index(self, index):
    return self.answer_list[index] == self.correct_answer
  