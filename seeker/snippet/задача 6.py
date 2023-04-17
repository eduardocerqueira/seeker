#date: 2023-04-17T16:43:11Z
#url: https://api.github.com/gists/9be14ad785d3f978ba6546a809b2a3f9
#owner: https://api.github.com/users/SergeyBadmaev

class Human:
    def __init__(self, name):
        self.name = name

    def answer_question(self, question):
        print('Очень интересный вопрос! Не знаю.')


class Student(Human):
    def __init__(self, name):
        self.name = name

    def ask_question(self, someone, question):
        print(f'{someone.name}, {question}')
        someone.answer_question(question)
        print()  # этот print выводит разделительную пустую строку	


class Curator(Human):
    def answer_question(self, question):
        if question == 'мне грустненько, что делать?':
            print('Держись, всё получится. Хочешь видео с котиками?')
        else:
            super().answer_question(question)

# объявите и реализуйте классы CodeReviewer и Mentor
class CodeReviewer(Human):
    def answer_question(self, question):
        if question == 'что не так с моим проектом?':
            print('О, вопрос про проект, это я люблю.')
        else:
            super().answer_question(question)
            
class Mentor(Human):
    def answer_question(self, question):
        if question == 'как устроиться работать питонистом?':
            print('Отдохни и возвращайся с вопросами по теории.')
        else:
            super().answer_question(question)            

# следующий код менять не нужно, он работает, мы проверяли
student1 = Student('Тимофей')
curator = Curator('Марина')
mentor = Mentor('Ира')
reviewer = CodeReviewer('Евгений')
friend = Human('Виталя')

student1.ask_question(curator, 'мне грустненько, что делать?')
student1.ask_question(mentor, 'мне грустненько, что делать?')
student1.ask_question(reviewer, 'когда каникулы?')
student1.ask_question(reviewer, 'что не так с моим проектом?')
student1.ask_question(friend, 'как устроиться на работу питонистом?')
student1.ask_question(mentor, 'как устроиться работать питонистом?')