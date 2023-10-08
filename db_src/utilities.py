import os
import sys
import time
import datetime
import json
import numpy as np
from scipy import stats
import pandas as pd
from multiprocessing import Pool
from functools import partial
from itertools import product
from tqdm import tqdm


def elapsed(t0):
	return np.around(time.time() - t0, 2)

def vrange(n, verbose=False):
	if not verbose:
		return range(n)
	else:
		return tqdm(list(range(n)))

def haslength(x):
	try:
		len(x)
		return True
	except:
		return False

### Multiprocessing helper
def _one_arg_function(list_of_inputs, args, func, kwargs):
	"""
	Globally-defined helper function for pickling in multiprocessing.
	:param list of inputs: List of inputs to a function
	:param args: Names/args for those inputs
	:param func: A function
	:param kwargs: Other kwargs to pass to the function. 
	"""
	new_kwargs = {}
	for i, inp in enumerate(list_of_inputs):
		new_kwargs[args[i]] = inp
	return func(**new_kwargs, **kwargs)

def apply_pool_factorial(
	func, 
	constant_inputs={}, 
	num_processes=1, 
	**kwargs
):
	"""
	Spawns num_processes processes to apply func to many different arguments.
	This wraps the multiprocessing.pool object plus the functools partial function. 
	
	Parameters
	----------
	func : function
		An arbitrary function
	constant_inputs : dictionary
		A dictionary of arguments to func which do not change in each
		of the processes spawned, defaults to {}.
	num_processes : int
		The maximum number of processes spawned, defaults to 1.
	kwargs : dict
		Each key should correspond to an argument to func and should
		map to a list of different arguments.
	Returns
	-------
	outputs : list
		List of outputs for each input, in the order of the inputs.
	Examples
	--------
	If we are varying inputs 'a' and 'b', we might have
	``apply_pool(
		func=my_func, a=[1,2], b=[5]
	)``
	which would return ``[my_func(a=1, b=5), my_func(a=2,b=5)]``.
	"""
	# Construct input sequence 
	args = sorted(kwargs.keys())
	kwarg_prod = list(product(*[kwargs[x] for x in args]))
	# Prepare to send this to apply pool
	final_kwargs = {}
	for i, arg in enumerate(args):
		final_kwargs[arg] = [k[i] for k in kwarg_prod]
	return apply_pool(
		func=func, 
		constant_inputs=constant_inputs,
		num_processes=num_processes,
		**final_kwargs
	)


def apply_pool(func, constant_inputs={}, num_processes=1, **kwargs):
	"""
	Spawns num_processes processes to apply func to many different arguments.
	This wraps the multiprocessing.pool object plus the functools partial function. 
	
	Parameters
	----------
	func : function
		An arbitrary function
	constant_inputs : dictionary
		A dictionary of arguments to func which do not change in each
		of the processes spawned, defaults to {}.
	num_processes : int
		The maximum number of processes spawned, defaults to 1.
	kwargs : dict
		Each key should correspond to an argument to func and should
		map to a list of different arguments.
	Returns
	-------
	outputs : list
		List of outputs for each input, in the order of the inputs.
	Examples
	--------
	If we are varying inputs 'a' and 'b', we might have
	``apply_pool(
		func=my_func, a=[1,3,5], b=[2,4,6]
	)``
	which would return ``[my_func(a=1, b=2), my_func(a=3,b=4), my_func(a=5,b=6)]``.
	"""

	# Construct input sequence
	args = sorted(kwargs.keys())
	num_inputs = len(kwargs[args[0]])
	for arg in args:
		if len(kwargs[arg]) != num_inputs:
			raise ValueError(f"Number of inputs differs for {args[0]} and {arg}")
	inputs = [[] for _ in range(num_inputs)]
	for arg in args:
		for j in range(num_inputs):
			inputs[j].append(kwargs[arg][j])

	# Construct partial function
	partial_func = partial(
		_one_arg_function, args=args, func=func, kwargs=constant_inputs,
	)

	# Don't use the pool object if num_processes=1
	num_processes = min(num_processes, len(inputs))
	if num_processes == 1:
		all_outputs = []
		for inp in inputs:
			all_outputs.append(partial_func(inp))
	else:
		with Pool(num_processes) as thepool:
			all_outputs = thepool.map(partial_func, inputs)

	return all_outputs

def create_output_directory(args, dir_type='misc', return_date=False):
	# Date
	today = str(datetime.date.today())
	hour = str(datetime.datetime.today().time())
	hour = hour.replace(':','-').split('.')[0]
	# Output directory
	file_dir = os.path.dirname(os.path.abspath(__file__))
	parent_dir = os.path.split(file_dir)[0]
	output_dir = f'{parent_dir}/sim_data/{dir_type}/{today}/{hour}/'
	# Ensure directory exists
	print(f"Output directory is {output_dir}")
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	# Save description
	args_path = output_dir + "args.json"
	with open(args_path, 'w') as thefile:
		thefile.write(json.dumps(args))
	# Return 
	if return_date:
		return output_dir, today, hour
	return output_dir


def compute_est_bounds(summands, alpha=0.05):
	"""
	Parameters
	----------
	summands : np.array
		(2, n)-shaped array

	Returns
	-------
	ests : np.array
		2-shaped array of lower and upper estimators (sample mean).
	bounds : np.array
		2-shaped array of lower/upper confidence bounds.
	"""
	ests = summands.mean(axis=1)
	ses = summands.std(axis=1) / np.sqrt(summands.shape[1])
	scale = stats.norm.ppf(1-alpha/2)
	return ests, np.array([
		ests[0] - scale * ses[0], ests[1] + scale * ses[1]
	])

class ConstantDist:

	def __init__(self, loc=0, scale=1):
		self.loc = loc
		self.scale = scale

	def rvs(self, size):
		return self.loc + self.scale * np.ones(size)

class BatchedCategorical:
	
	def __init__(
		self, vals, probs
	):
		"""
		Parameters
		----------
		vals : (n, nvals)-shaped array
		probs : (n, nvals)-shaped array. probs.sum(axis=1) == 1.
		"""
		inds = np.argsort(vals, axis=1)
		self.n, self.nvals = vals.shape
		self.vals = np.take_along_axis(vals, inds, axis=1)
		self.probs = np.take_along_axis(probs, inds, axis=1)
		self.cumprobs = np.cumsum(self.probs, axis=1)
		# validate args
		if np.any(self.probs < 0):
			raise ValueError("probs must be nonnegative")
		if np.any(np.abs(self.probs.sum(axis=1)-1) > 1e-5):
			raise ValueError("probs.sum(axis=1) must equal 1")
	
	def mean(self):
		return np.sum(self.vals * self.probs, axis=1)
	
	def ppf(self, q):
		"""
		q : (m,n)-shaped array.
		"""
		m = q.shape[0]
		if len(q.shape) == 1:
			q = np.stack([q for _ in range(self.n)], axis=1)
		# use a for loop to save memory
		qvals = np.zeros((m, self.n))
		for i in range(self.n):
			flags = self.cumprobs[i].reshape(self.nvals, 1) >= q[:, i]
			qvals[:, i] = self.vals[i][np.argmax(flags, axis=0)]
		return qvals

def parse_dist(dist, df=4, loc=0, scale=1, req_symmetric=False, **kwargs):
	# sometimes return a regular dist object
	if not isinstance(dist, str):
		return dist

	# Parse
	dist = dist.lower()
	if dist == 'constant':
		return ConstantDist(loc=loc, scale=scale)
	if dist == 'gaussian':
		return stats.norm(loc=loc, scale=scale)
	if dist == 'invchi2':
		return stats.chi2(df=df, loc=loc, scale=scale/df)
	if dist == 't' or dist == 'tdist':
		return stats.t(loc=loc, scale=scale, df=df)
	if dist == 'cauchy':
		return stats.cauchy(loc=loc, scale=scale)
	if dist == 'laplace':
		return stats.laplace(loc=loc, scale=scale)
	if dist in ['uniform', 'unif']:
		return stats.uniform(loc=loc-scale/2, scale=scale)
	# warning for certain cases
	if req_symmetric:
		raise ValueError(
			f"Dist {dist} is not a recognized symmetric distribution."
		)
	if dist in ['expo', 'expon', 'exponential']:
		return stats.expon(loc=loc-scale, scale=scale)
	if dist == 'gamma':
		a = kwargs.pop("a", 5)
		return stats.gamma(loc=loc-a*scale, scale=scale, a=a)
	else:
		raise ValueError(f"Dist {dist} is unrecognized.")