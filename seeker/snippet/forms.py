#date: 2023-03-03T16:53:49Z
#url: https://api.github.com/gists/18a881dacbb0d0c902f5e218228f2024
#owner: https://api.github.com/users/docsallover

from django import forms

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()