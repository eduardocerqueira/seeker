#date: 2023-06-16T17:01:47Z
#url: https://api.github.com/gists/27000568553db5f27bfd29e92bb1072d
#owner: https://api.github.com/users/JamesTheAwesomeDude

import greenlet


# TODO: address consumers that exit before consuming the whole iterable?
# TODO: address (with alternate function) generator-function consumers? (NOTE: if they yield not in 1-to-1 correspondence with consumption, even defining the input behavior may be *painful*)

def teeinto_lockstep(
	     it: 'Iterable[TypeVar("T")]',
	     *consumer_callables: 'Callable[[Iterable[TypeVar("T")]], Any]'
	) -> 'Iterable[TypeVar("T")]':
	"tee unbounded iterable *it* into more than one dumb consumer function."
	# h/t https://morestina.net/blog/1378/parallel-iteration-in-python for inspiration

	_stop = object()
	_gyield = lambda value=None, *, _parent=greenlet.getcurrent(): _parent.switch(value)

	# 1. Create greenlets and start consumers
	green_consumers = []
	for consumer_callable in consumer_callables:
		green_consumer = greenlet.greenlet(consumer_callable)
		green_consumer.switch(iter(_gyield, _stop))  # Start consumer
		green_consumers.append(green_consumer)

	# 2. Iterate
	for value in it:
		for green_consumer in green_consumers:
			# When .switch() is called, the iter object's __next__() RETURNS,
			# allowing the consumer's for-loop (or whatever) to advance once.
			green_consumer.switch(value)
		yield value

	# 3. Propagate StopIteration to consumers
	for green_consumer in green_consumers:
		green_consumer.switch(_stop)
