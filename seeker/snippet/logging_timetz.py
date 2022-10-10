#date: 2022-10-10T17:06:11Z
#url: https://api.github.com/gists/379bf88c63abc2e359b6f6072a5ee7d2
#owner: https://api.github.com/users/jongphago

def timetz(*args):
    return datetime.now(timezone('Asia/Seoul')).timetuple()
