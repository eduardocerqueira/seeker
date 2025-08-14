#date: 2025-08-14T16:51:18Z
#url: https://api.github.com/gists/86d0c3f63363cbc9b818f9a07ff871d4
#owner: https://api.github.com/users/20100dbg

#!/usr/bin/env python3

from queue import Queue
import threading
import time


class My_Worker():

	def __init__(self, nb_threads = 10):
		self.nb_threads = nb_threads
		self.queue = Queue()
		self.done = threading.Event()

	def watcher(self):
		self.queue.join()
		self.done.set()

	#Can dynamically add tasks
	def add_work(self, task):
		self.queue.put(task)

	def start(self, tasks):
		for task in tasks:
			self.add_work(task)

		print(f"{len(tasks)} tasks to do")
		
		workers = [threading.Thread(target=self.worker,args=[],daemon=True) for worker_id in range(self.nb_threads)]
		[worker.start() for worker in workers]
		threading.Thread(target=self.watcher,args=(),daemon=True).start()
		
		try:
			self.done.wait()
		except KeyboardInterrupt:
			print("\nTerminating...")


	#do the actual work
	def worker(self):
		while True:
			task = self.queue.get()

			if task is None:
				#no more work
				self.queue.task_done()
				continue
			else:
				#do stuff
				time.sleep(0.5)

				print(f"{task} done")
				self.queue.task_done()


#Init worker with number of threads
worker = My_Worker(2)

#Start worker with array of tasks
worker.start(range(10))

#Or
#for _ in range(3):
#	worker.add_work(_)
#worker.start()