#date: 2023-05-22T17:04:25Z
#url: https://api.github.com/gists/3a0ed7480a117eac7262d3d2f84e2226
#owner: https://api.github.com/users/webremake

import types
from functools import wraps


def humanify(name: str):
    import re
    return ' '.join(re.split('_+', name))


def step(fn):
    @wraps(fn)
    def fn_with_logging(*args, **kwargs):
        is_method = (
                args
                and isinstance(args[0], object)
                and isinstance(getattr(args[0], fn.__name__), types.MethodType)
        )

        args_to_log = args[1:] if is_method else args
        args_and_kwargs_to_log_as_strings = [
            *map(str, args_to_log),
            *[f'{key}={value}' for key, value in kwargs.items()]
        ]
        args_and_kwargs_string = (
            (': ' + ', '.join(map(str, args_and_kwargs_to_log_as_strings)))
            if args_and_kwargs_to_log_as_strings
            else ''
        )

        print(
            (f'[{args[0].__class__.__name__}] ' if is_method else '')
            + humanify(fn.__name__)
            + args_and_kwargs_string
        )

        return fn(*args, **kwargs)

    return fn_with_logging


'''
def given_sign_up_form_opened():
    log_step(given_sign_up_form_opened.__name__)
    ...
given_sign_up_form_opened = step(given_sign_up_form_opened)
'''


@step
def given_sign_up_form_opened():
    ...


class SignUpForm:

    @step
    def fill_name(self, first_name, surname):
        pass
        return self

    @step
    def fill_email(self, value):
        pass
        return self

    @step
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"f "**********"i "**********"l "**********"l "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"v "**********"a "**********"l "**********"u "**********"e "**********") "**********": "**********"
        pass
        return self

    @step
    def submit(self):
        pass
        return self


class DashBoard:
    ...

    @step
    def go_to_user_profile(self):
        pass


sign_up_form = SignUpForm()
dashboard = DashBoard()

given_sign_up_form_opened()
(sign_up_form
 .fill_name('yasha', surname='Kramarenko')
 .fill_email('yashaka@gmail.com')
 .fill_password('qwerty')
 .submit()
 )
dashboard.go_to_user_profile()

'''
# will log to console the following:

given sign up form opened
[SignUpForm] fill name: yasha, surname=Kramarenko
[SignUpForm] fill email: yashaka@gmail.com
[SignUpForm] fill password: "**********"
[SignUpForm] submit
[DashBoard] go to user profile

Process finished with exit code 0
'''
