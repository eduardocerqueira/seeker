#date: 2024-02-21T17:03:05Z
#url: https://api.github.com/gists/789c7f6dc2a7f793e1b85888e39fd0a9
#owner: https://api.github.com/users/mvandermeulen

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FileName  : redistools.py
# Author    : wuqingfeng@

from functools import wraps
try:
    import cPickle as pickle
except ImportError:
    import pickle
from uuid import uuid4
import uuid
import time
import threading
import redis

__all__ = ['RedisClient', 'HotCache', 'HotQueue', 'HotPub', 'HotSub']


def convert_kwargs(kwargs):
    new_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(k, str):
            new_kwargs[k.lower()] = v
    return new_kwargs


class TimeoutError(Exception):
    pass


class RedisClient():

    """
    Singleton pattern
    http://stackoverflow.com/questions/42558/python-and-the-singleton-pattern
    """

    __instance = {}
    __rclis = {}

    def __new__(cls, **kwargs):
        kwargs = convert_kwargs(kwargs)
        if str(kwargs) not in cls.__instance:
            if 'url' in kwargs:
                pool = redis.ConnectionPool.from_url(**kwargs)
            else:
                pool = redis.ConnectionPool(**kwargs)
            cls.__rclis[str(kwargs)] = redis.StrictRedis(connection_pool=pool)
            cls.__instance[str(kwargs)] = super(RedisClient, cls).__new__(cls)
        return cls.__instance[str(kwargs)]

    def __init__(self, **kwargs):
        kwargs = convert_kwargs(kwargs)
        self.rcli = self.__rclis[str(kwargs)]

    @property
    def __dict__(self):
        try:
            return self.rcli.__dict__
        except RuntimeError:
            raise AttributeError('__dict__')

    def __dir__(self):
        try:
            return dir(self.rcli)
        except RuntimeError:
            return []

    def __getattr__(self, name):
        if name == '__members__':
            return dir(self.rcli)
        return getattr(self.rcli, name)


def key_for_name(name):
    return 'queue:%s' % name


def result_for_name(taskid):
    return 'result:%s' % taskid


def get_mac_address(): 
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:] 
    return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])


class HotCache():
    """
    Args:
        serializer:
            the class or module to serialize msgs with, must have
            methods or functions named ``dumps`` and ``loads``
            such as `pickle` and json are good idea
    """

    def __init__(self, name, serializer=pickle, **kwargs):
        self.serializer = serializer
        self.__redis = RedisClient(**kwargs)

    def set(self, name, msgs, timeout=None):
        if self.serializer is not None:
            msgs = self.serializer.dumps(msgs)
        self.__redis.set(name, msgs)
        if timeout is not None:
            self.__redis.expire(name, int(timeout))

    def add(self, name, msgs, timeout=None):
        if self.__redis.exists(name):
            return False
        else:
            self.set(name, msgs, timeout)
            return True

    def get(self, name):
        msgs = self.__redis.get(name)
        try:
            if msgs is not None and self.serializer is not None:
                return self.serializer.loads(msgs)
        except:
            pass
        return msgs

    def isexist(self, name):
        if self.__redis.exists(name):
            return True
        else:
            return False

    def delete(self, name):
        if self.__redis.exists(name):
            self.__redis.delete(name)


class HotQueue():

    """Simple FIFO message queue stored in a Redis list. Example:
    >>> from hotqueue import HotQueue
    >>> queue = HotQueue("myqueue", host="localhost", port=6379, db=0)
    
    :param name: name of the queue
    :param serializer: the class or module to serialize msgs with, must have
        methods or functions named ``dumps`` and ``loads``,
        `pickle <http://docs.python.org/library/pickle.html>`_ is the default,
        use ``None`` to store messages in plain text (suitable for strings,
        integers, etc)
    :param kwargs: additional kwargs to pass to :class:`Redis`, most commonly
        :attr:`host`, :attr:`port`, :attr:`db`
    """

    def __init__(self, name, serializer=pickle, **kwargs):
        self.name = name
        self.serializer = serializer
        self.__redis = RedisClient(**kwargs)

    def __len__(self):
        return self.__redis.llen(self.key)

    @property
    def key(self):
        return key_for_name(self.name)

    def clear(self):
        """Clear the queue of all messages, deleting the Redis key."""
        self.__redis.delete(self.key)

    def consume(self, **kwargs):
        """Return a generator that yields whenever a message is waiting in the
        queue. Will block otherwise. Example:
        
        >>> for msg in queue.consume(timeout=1):
        ...     print msg
        my message
        another message
        
        :param kwargs: any arguments that :meth:`~hotqueue.HotQueue.get` can
            accept (:attr:`block` will default to ``True`` if not given)
        """

        kwargs.setdefault('block', True)
        try:
            while True:
                msg = self.get(**kwargs)
                if msg is None:
                    break
                yield msg
        except KeyboardInterrupt:
            print
            return

    def get(self, block=False, timeout=None):
        """Return a message from the queue. Example:
    
        >>> queue.get()
        'my message'
        >>> queue.get()
        'another message'
        
        :param block: whether or not to wait until a msg is available in
            the queue before returning; ``False`` by default
        :param timeout: when using :attr:`block`, if no msg is available
            for :attr:`timeout` in seconds, give up and return ``None``
        """

        if block:
            if timeout is None:
                timeout = 0
            msg = self.__redis.blpop(self.key, timeout=timeout)
            if msg is not None:
                msg = msg[1]
        else:
            msg = self.__redis.lpop(self.key)
        if msg is not None and self.serializer is not None:
            msg = self.serializer.loads(msg)
        return msg

    def put(self, *msgs):
        """Put one or more messages onto the queue. Example:
        
        >>> queue.put("my message")
        >>> queue.put("another message")
        
        To put messages onto the queue in bulk, which can be significantly
        faster if you have a large number of messages:
        
        >>> queue.put("my message", "another message", "third message")
        """

        if self.serializer is not None:
            msgs = map(self.serializer.dumps, msgs)
        self.__redis.rpush(self.key, *msgs)

    def publish(self, *msgs):
        """Publish one or more messages onto the queue for worker. Example:
        
        >>> queue.publish("my message")
        >>> queue.publish("another message")
        
        To put messages onto the queue in bulk, which can be significantly
        faster if you have a large number of messages:
        
        >>> queue.publish("my message", "another message", "third message")
        """
        taskid = str(uuid4())
        resultname = result_for_name(taskid)
        if self.serializer is not None:
            resultname = self.serializer.dumps(resultname)
            msgs = map(self.serializer.dumps, msgs)
        self.__redis.rpush(self.key, resultname, *msgs)
        return taskid

    def getresult(self, taskid, waittime=0.5, delflag=False):
        """
        get the result of the worker's return by taskid
        >>> result = queue.getresult("c1925dc0-48c3-4b40-a7f5-b5b9291663a8")
        >>> print result
        >>> "hello world"
        
        if can't find result, return None
        """
        resultname = result_for_name(taskid)

        if waittime:
            time.sleep(waittime)
        
        result = self.__redis.get(resultname)

        if result is not None and self.serializer is not None:
            result = self.serializer.loads(result)
        else:
            result = None

        if delflag and self.__redis.exists(resultname):
            self.__redis.delete(resultname)

        return result

    def worker(self, *args, **kwargs):
        """Decorator for using a function as a queue worker. Example:
        
        >>> @queue.worker(timeout=1)
        ... def printer(msg):
        ...     print msg
        >>> printer()
        my message
        another message
        
        You can also use it without passing any keyword arguments:
        
        >>> @queue.worker
        ... def printer(msg):
        ...     print msg
        >>> printer()
        my message
        another message
        
        :param kwargs: any arguments that :meth:`~hotqueue.HotQueue.get` can
            accept (:attr:`block` will default to ``True`` if not given)
        """

        def decorator(worker):
            @wraps(worker)
            def wrapper(*args):
                resultname = None
                # resultlist = []
                if 'resultexpire' in kwargs:
                    resultexpire = kwargs.pop('resultexpire')
                else:
                    resultexpire = 60 * 60
                for msg in self.consume(**kwargs):
                    # print msg
                    if 'result:' in msg and not resultname:
                        resultname = msg
                    else:
                        result = worker(*args + (msg,))
                        if self.serializer is not None and resultname is not None:
                            pre_result = self.__redis.get(resultname)
                            if pre_result is not None:
                                pre_result = self.serializer.loads(pre_result)
                                if isinstance(pre_result, list):
                                    pre_result.append(result)
                                    result = pre_result
                                else:
                                    result = [pre_result, result]
                            self.__redis.set(resultname, self.serializer.dumps(result), ex=resultexpire)
            return wrapper

        if args:
            return decorator(*args)

        return decorator


class HotPub():

    def __init__(self, channel, serializer=pickle, **kwargs):
        self.channel = channel
        self.serializer = serializer
        self.__redis = RedisClient(**kwargs)
        self.taskid = None

    @property
    def key(self):
        return key_for_name(self.channel)

    def publish(self, *msgs):
        """Publish one or more messages onto the queue. Example:
        
        >>> pub.publish("my message")
        >>> pub.publish("another message")
        
        To put messages onto the queue in bulk, which can be significantly
        faster if you have a large number of messages:
        
        >>> pub.publish("my message", "another message", "third message")
        """

        self.taskid = str(uuid4())
        resultname = result_for_name(self.taskid)
        
        if self.serializer is not None:
            for msg in msgs:
                self.__redis.publish(self.channel, self.serializer.dumps({'resultname': resultname, 'message': msg}))

        return self.taskid

    def get(self, timeout=None, vlist=True, delflag=False, threadmode=0):

        def workthread(resultdata=[], vlist=True, delflag=False, maxretry=200):
            resultname = result_for_name(self.taskid)
            result = {}
            # result = self.__redis.hgetall(resultname)
            count = 0
            while not result:
                if count > maxretry:
                    break
                time.sleep(0.005)
                lockname = resultname + 'lock'
                with self.__redis.lock(lockname):
                    if self.__redis.exists(resultname):
                        result = self.__redis.hgetall(resultname)
                count += 1
            if vlist:
                results = []
                if result and self.serializer is not None:
                    for subvalue in result.values():
                        results.append(self.serializer.loads(subvalue))
                else:
                    results = result.values()
            else:
                results = {}
                if result and self.serializer is not None:
                    for subname, subvalue in result.items():
                        results[subname] = self.serializer.loads(subvalue)
                else:
                    results = result
            
            if delflag and self.__redis.exists(resultname):
                self.__redis.delete(resultname)
            resultdata.append(results)
        
        resultdata = []
        kwargs = {
            'resultdata': resultdata,
            'vlist': vlist,
            'delflag': delflag
        }

        if timeout:
            kwargs["maxretry"] = timeout / 0.005

        if threadmode:
            worker = threading.Thread(target=workthread, args=(), kwargs=kwargs)
            worker.setDaemon(True)
            worker.start()
            if timeout:
                worker.join(timeout + 1)
            else:
                worker.join()
            alive = worker.isAlive()
            if alive:
                raise TimeoutError('get taskid: %s result is timeout!' % self.taskid)
            else:
                return resultdata[0]
        else:
            workthread(**kwargs)
            return resultdata[0]

    def getresult(self, taskid, waittime=1, vlist=True, delflag=False):
        resultname = result_for_name(taskid)

        if waittime:
            time.sleep(waittime)
        
        result = self.__redis.hgetall(resultname)

        if vlist:
            results = []
            if result and self.serializer is not None:
                for subvalue in result.values():
                    results.append(self.serializer.loads(subvalue))
            else:
                results = result.values()
        else:
            results = {}
            if result and self.serializer is not None:
                for subname, subvalue in result.items():
                    results[subname] = self.serializer.loads(subvalue)
            else:
                results = result
        
        if delflag and self.__redis.exists(resultname):
            self.__redis.delete(resultname)

        return results


class HotSub():
    def __init__(self, channels=[], name=None, serializer=pickle, **kwargs):
        self.channels = {}
        self.serializer = serializer
        self.name = name or get_mac_address()
        self.__redis = RedisClient(**kwargs)
        self.__sub = self.__redis.pubsub(ignore_subscribe_messages=True)
        if channels is list:
            self.__sub.subscribe(*channels)
            self.channels = dict.fromkeys(channels)

    @property
    def key(self):
        return key_for_name(self.channels.keys())

    def psubscribe(self, channels=[]):
        self.__sub.psubscribe(*channels)
        self.channels.update(dict.fromkeys(channels))

    def get(self):
        msg = self.__sub.get_message()
        if msg is not None:
            # msg = self.serializer.loads(msg)
            data = msg.get('data')
            if data is not None and self.serializer is not None:
                data = self.serializer.loads(data)
                # resultname = data.get('resultname')
                message = data.get('message')
                msg['data'] = message
        return msg

    def consume(self):
        try:
            while True:
                msg = self.get()
                if msg is None:
                    break
                yield msg
        except KeyboardInterrupt:
            print
            return

    def clear(self, channels=[], type=None):
        """Clear the channel of the channels, or clear all channels"""
        if channels:
            if type is None:
                self.__sub.unsubscribe(*channels)
            else:
                self.__sub.punsubscribe(*channels)
            for channel in channels:
                del self.channels[channel]
        else:
            self.__sub.unsubscribe(*channels)
            self.__sub.punsubscribe(*channels)
            self.channels = {}

    def worker(self, *args, **kwargs):
        """Decorator for using a function as a subscribe worker. Example:
        msg is a required argument to accept the msg from publisher
        >>> @sub.worker(channel="my-first-channel")
        ... def printer(msg):
        ...     print msg
        >>> printer()
        my message
        another message
        """
        def decorator(worker):
            @wraps(worker)
            def wrapper():
                channel = kwargs.get("channel")
                sleep = kwargs.get("sleep") or 2
                resultexpire = kwargs.get('resultexpire') or 60 * 60
                if channel is not None:
                    def wrapper_worker(msg):
                        resultname = None
                        data = msg.get('data')
                        if data is not None and self.serializer is not None:
                            data = self.serializer.loads(data)
                            resultname = data.get('resultname')
                            message = data.get('message')
                            msg['data'] = message
                            result = worker(msg)
                            if resultname is not None:
                                pre_result = self.__redis.hget(resultname, self.name)
                                if pre_result is not None:
                                    pre_result = self.serializer.loads(pre_result)
                                    if isinstance(pre_result, list):
                                        pre_result.append(result)
                                        result = pre_result
                                    else:
                                        result = [pre_result, result]
                                lockname = resultname + 'lock'
                                with self.__redis.lock(lockname):
                                    self.__redis.hset(resultname, self.name, self.serializer.dumps(result))
                                    self.__redis.expire(resultname, resultexpire)

                    self.__sub.subscribe(**{channel: wrapper_worker})
                    thread = self.__sub.run_in_thread(sleep_time=sleep)
                    return thread
                else:
                    return None
            return wrapper
        if args:
            return decorator(*args)
        return decorator


if __name__ == "__main__":
    """
    this is a example of the HotPub and HotSub, python redistools to run and test it
    """
    import os
    from yaml import load
    try:
        from yaml import Cloader as Loader
    except ImportError:
        from yaml import Loader

    config_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config.yaml'))
    config = load(file(config_path, 'r'), Loader=Loader)
    redis_server = config['REDIS']
    # print redis_server
    # rcli = RedisClient(**redis_server)
    # rcli.set('foo', 'bar')
    # print rcli.get('foo')
    
    # queue = HotQueue('mytestqueue', **redis_server)
    # taskid = queue.publish({'a': 1, 'b': 2}, {'a': 5, 'b': 6})

    # print taskid

    # time.sleep(60)

    # @queue.worker
    # def adder(msg):
    #     # print msg
    #     a = msg.get('a', 0)
    #     b = msg.get('b', 0)
    #     return a+b

    # adder()

    # result = queue.getresult(taskid)

    # print result

    sub = HotSub(**redis_server)

    @sub.worker(channel="my-first-channel")
    def adder(msg):
        data = msg.get('data')
        a = data.get("a")
        b = data.get("b")
        # time.sleep(10)
        return a + b

    thread = adder()
    pub = HotPub(channel="my-first-channel", **redis_server)
    while True:
        try:
            taskid = pub.publish({"a": 1, "b": 2}, {"a": 2, "b": 5})
            print('taskid', taskid)
            result = pub.get(timeout=5)
            print(result)
            time.sleep(10)
        except KeyboardInterrupt:
            thread.stop()
            break
