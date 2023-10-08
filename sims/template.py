"""
Template for running simulations.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from context import src
from db_src import parser, utilities

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

def single_seed_sim(
	**args
):
	print(args)

def main(args):
	# Parse arguments
	args = parser.parse_args(args)
	reps = args.get('reps', [1])[0]
	num_processes = args.get('num_processes', [1])[0]

	# Key defaults go here
	args['tau'] = args.get("tau", [0])
	
	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	args.pop("description")

	# Run outputs
	outputs = utilities.apply_pool_factorial(
		func=single_seed_sim,
		seed=list(range(1, reps+1)), 
		num_processes=num_processes, 
		**args,
	)



if __name__ == '__main__':
	main(sys.argv)