#!/bin/bash

REPS=10
NUM_PROCESSES=5

ARGS_HOMOSKED="
	--p 20
	--kappa [5,10,30,50]
	--eps_dist gaussian
	--reps $REPS
	--num_processes $NUM_PROCESSES
	--tau 2
	--r2 0.9
	--betaS_norm 0
"

ARGS_HETEROSKED="
	${ARGS_HOMOSKED}
	--heterosked norm
	--tauv [0.2,5]
"

python3.9 lee_sims.py $ARGS_HOMOSKED
python3.9 lee_sims.py $ARGS_HETEROSKED