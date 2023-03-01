#date: 2023-03-01T17:03:26Z
#url: https://api.github.com/gists/ad6db746a43195522fdeaaa954ea4854
#owner: https://api.github.com/users/Viber-Air

class Stats:
    def _init_(self, broker='kafka', port='9092', topic='stats'):
        conf = {'bootstrap.servers': f"{broker}",'client.id': socket.gethostname()}
        self.client = socket.gethostname()
        self.producer = Producer(conf)
        self.topic = topic
        self.flush_thread = Thread(target=self.flush)
        self.flushing = True
        self.flush_thread.start()
        
    def flush(self, period=5):
        while self.flushing:
            self.producer.flush()
            sleep(period)

    def timeit(self, func):
        ''' Keep track of function/method execution time.
            WARNING: it doesn't track if exception occured'''
        def inner(*args, **kwargs):
            t_init = time()
            resp = func(*args, **kwargs)
            t_end = time()
            message = {
                'method':'timeit',
                'file_name': _name_,
                'function_name': func._name_,
                'service_name': self.client,
                'exec_time': t_end - t_init
            }
            message = json.dumps(message).encode()
            self.producer.produce(self.topic, value=message)
            return resp
        return inner
        
  stats = Stats(broker=f'{KAFKA_HOST}:{KAFKA_PORT}')