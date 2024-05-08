#date: 2024-05-08T16:39:30Z
#url: https://api.github.com/gists/a15c005df3c524ea9d049c8f5e03ffbb
#owner: https://api.github.com/users/vikramsoni2

from logging.config import ConvertingList, ConvertingDict, valid_ident
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
import atexit
import logging


def _resolve_handlers(l):
    if not isinstance(l, ConvertingList):
        return l

    # Indexing the list performs the evaluation.
    resolved_handlers = []
    for i in range(len(l)):
        handler = l[i]
        if isinstance(handler, dict) and handler.get('class') == 'logging.StreamHandler':
            # Create the StreamHandler manually
            stream_handler = logging.StreamHandler(stream=handler['stream'])
            stream_handler.setLevel(handler.get('level', logging.NOTSET))
            if 'formatter' in handler:
                stream_handler.setFormatter(handler['formatter'])
            handler = stream_handler
        resolved_handlers.append(handler)
    return resolved_handlers


def _resolve_queue(q):
    if not isinstance(q, ConvertingDict):
        return q
    if '__resolved_value__' in q:
        return q['__resolved_value__']

    cname = q.pop('class')
    klass = q.configurator.resolve(cname)
    props = q.pop('.', None)
    kwargs = {k: q[k] for k in q if valid_ident(k)}
    result = klass(**kwargs)
    if props:
        for name, value in props.items():
            setattr(result, name, value)

    q['__resolved_value__'] = result
    return result



class QueueListenerHandler(QueueHandler):

    def __init__(self, handlers, respect_handler_level=False, auto_run=True, queue=Queue(-1)):
        
        queue = _resolve_queue(queue)
        super().__init__(queue)
        handlers = _resolve_handlers(handlers)
        
        self._listener = QueueListener(
            self.queue,
            *handlers,
            respect_handler_level=respect_handler_level)
        if auto_run:
            self.start()
            atexit.register(self.stop)

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def emit(self, record):
        return super().emit(record)
        



LOGGING_CONFIG = {
    'version': 1,
    'objects':{
        'queue': {
            'class': 'queue.Queue',
            'maxsize': 1000
        }
    },
    'formatters': {
        'json': {
          '()' : 'pythonjsonlogger.jsonlogger.JsonFormatter',
          'format': '%(asctime)s %(name)-12s %(levelname)-4s %(message)s',
        }
    },
    'handlers': {
        'stream': {
            'class': 'logging.StreamHandler',
            'formatter': 'json',
            'level': logging.INFO,
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': logging.INFO,
            'formatter': 'json',
            'filename': 'logs/application.log',
            'mode': 'a',
        },
        'queue_listener':{
            '()': QueueListenerHandler,
            'handlers':[
                'cfg://handlers.stream',
                'cfg://handlers.file'
            ],
            'queue': 'cfg://objects.queue'
        }  
    },
    'loggers': {
        '': {
            'handlers': ['stream', 'file'],
            'level': logging.INFO,
            'propagate': False
        }
    }
}


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)