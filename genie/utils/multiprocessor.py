import os
import random 
import math
from multiprocessing import Process
from abc import ABC, abstractmethod


class MultiProcessor(ABC):
	"""
	Base class for multiprocessing.
	"""

	@abstractmethod
	def create_tasks(self, params):
		"""
		Define a list of tasks to be distributed across processes, where each 
		task is defiend as a dictionary of task-specific parameters.

		Args:
			params:
				A dictionary of parameters.

		Returns:
			A list of tasks, where each task is defiend as a dictionary of 
			task-specific parameters.
		"""
		raise NotImplemented

	@abstractmethod
	def create_constants(self, params):
		"""
		Define a dictionary of constants shared across processes.

		Args:
			params:
				A dictionary of parameters.

		Returns:
			A dictionary of constants shared across processes.
		"""
		raise NotImplemented

	@abstractmethod
	def execute(self, constants, tasks, device):
		"""
		Execute a list of tasks on the given device.

		Args:
			constants:
				A dictionary of constants.
			tasks:
				A list of tasks, where each task is defiend as a dictionary 
				of task-specific parameters.
			device:
				Device to run on.
		"""
		raise NotImplemented

	def run(self, params, num_devices, sequential_order=False):
		"""
		Run in parallel based on input parameters/configurations.

		Args:
			params:
				A dictionary of parameters/configurations.
			num_devices:
				Number of GPUs availble.
			sequential_order:
				Flag on whether to shuffle tasks.
		"""


		# Create tasks
		tasks = self.create_tasks(params)

		# Balance load of devices
		if num_devices > 1 and not sequential_order:
			random.shuffle(tasks)

		# Create constants
		constants = self.create_constants(params)

		# Start parallel processes
		processes = []
		binsize = math.ceil(len(tasks) / num_devices)
		for i in range(num_devices):
			p = Process(
				target=self.execute,
				args=(
					constants,
					tasks[binsize*i:binsize*(i+1)],
					f'cuda:{i}'
				)
			)
			p.start()
			processes.append(p)

		# Wait for completion
		for p in processes:
			p.join()
	