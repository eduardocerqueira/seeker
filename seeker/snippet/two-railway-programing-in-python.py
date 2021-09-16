#date: 2021-09-16T17:03:07Z
#url: https://api.github.com/gists/b7b1c05a840716604f9b75e55e8dd8f5
#owner: https://api.github.com/users/jeffersonfragoso

class Success(object):
    def __init__(self, value):
        self.value = value

class Error(object):
    def __init__(self, value):
        self.value = value

class wrapper(object):
    def __init__(self, result):
        self.result = result

    def success(self, func):
        if isinstance(self.result, Success):
            func(self.result.value)
        return self

    def fail(self, func):
        if isinstance(self.result, Error):
            func(self.result.value)
        return self

def compose(*func):
    def f(x):
        r = Success(x)
        for f in func:
            if isinstance(r, Error): break
            r = f(r.value)
        return wrapper(r)
    return f


def validate(data):
    if data.get('f', None):
        return Success(data)
    return Error({'error': 'field f required'})

def create_user(data):
    print('create user named ' + data['f'])
    return Success(data)

def send_email(data):
    try:
        raise IOError("failed to send error")
    except IOError as e:
        return Error(e)


req = {'f': '1'}

f = compose(validate, create_user, send_email)
def success(r):
    print("Success, ", r)
def fail(r):
    print("Failed: ", r)

f(req).success(success).fail(fail)
