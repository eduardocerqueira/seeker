#date: 2023-11-17T16:59:20Z
#url: https://api.github.com/gists/7dd789eb85a326b966f95d232047f05b
#owner: https://api.github.com/users/LF1scher

def confirm():
    answer = input("[Y/N]?")
    if answer.lower() in ["y","yes",""]:
        return True
    return False