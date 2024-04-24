#date: 2024-04-24T17:04:23Z
#url: https://api.github.com/gists/d9c1d079441b7950b1c53a86a8a12c71
#owner: https://api.github.com/users/sam-borb

# Import required modules from Django.
# 'settings' module for configuration of the Django settings.
# 'HttpResponse' class for sending the response to the user.
# 'path' function for defining a url path.
# 'get_wsgi_application' function for setting up default WSGI settings.
# 'execute_from_command_line' function for managing Django from the command line.
from django.conf import settings
from django.http import HttpResponse
from django.urls import path
from django.core.wsgi import get_wsgi_application
from django.core.management import execute_from_command_line

# Configure Django settings for this application.
# 'ROOT_URLCONF' is set to the current module,
# 'DEBUG' is set to True, so we get detailed error messages.
# 'SECRET_KEY' is set to a not-so-secret constant string.
settings.configure(
    ROOT_URLCONF=__name__,
    DEBUG=True,
    SECRET_KEY= "**********"
)

# Define a simple view function.
# Takes an HttpRequest object and returns an HttpResponse object.
# This view returns the text "Hello, world!".
def index(request):
    return HttpResponse("Hello, world!")

# Define the urlpatterns list, which routes urls to views.
# In this case, the root url is routed to the index view.
urlpatterns = [
    path('', index),
]

# If this module is being run as the main script,
# execute the management command from the command line.
# Run the file: python server.py
if __name__ == "__main__":
    execute_from_command_line(["manage.py", "runserver", "0.0.0.0:8000"])
