#date: 2022-06-17T17:02:31Z
#url: https://api.github.com/gists/471bc58e08abcddfb2d6229a40b5aa9a
#owner: https://api.github.com/users/katrbhach

from app import app


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
