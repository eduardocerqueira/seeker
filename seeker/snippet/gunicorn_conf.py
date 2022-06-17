#date: 2022-06-17T17:02:31Z
#url: https://api.github.com/gists/471bc58e08abcddfb2d6229a40b5aa9a
#owner: https://api.github.com/users/katrbhach

bind = "0.0.0.0:8080"
workers = 1
worker_class = 'gevent'
accesslog = "-"
access_log_format = \
    '%(t)s [INFO] completed %(m)s call to [%(U)s] with query parameters [%(q)s] in %(M)d ms. ' \
    'resultant status code is %(s)s"'
loglevel = 'error'
