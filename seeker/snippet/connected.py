#date: 2022-04-04T17:16:02Z
#url: https://api.github.com/gists/a56f33dca72716776e596fe1617977ce
#owner: https://api.github.com/users/yazeedabdulhai

from kivy.app import App
from kivy.uix.screenmanager import Screen, SlideTransition

class Connected(Screen):
    def disconnect(self):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = 'login'
        self.manager.get_screen('login').resetForm()
