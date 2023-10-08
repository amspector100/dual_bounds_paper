"""
Template for running simulations.
"""

import os
import sys
import time

import numpy as np
from scipy import stats
import pandas as pd
import cvxpy as cp
from context import db_src
from db_src import parser, utilities, gen_data, lee
from db_src.utilities import elapsed

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]
DEFAULT_NFOLDS = 5

COLUMNS = [
	'seed',
	'n',
	'p', 
	'eps_dist',
	'lmda_dist',
	'heterosked',
	'r2',
	'betaW_norm',
	'betaS_norm',
	'covmethod',
	'tau',
	'tauv',
	'stau',
	'method',
	'lower_est',
	'upper_est',
	'lower_bound',
	'upper_bound',
	'runtime',
]

GROUP_VALS = [    
	'n',
	'p', 
	'eps_dist',
	'heterosked',
	'r2',
	'covmethod',
	'tau',
	'stau',
	'tauv',
	'method',
]

ESTIMAND_COLUMNS = [
	'n',
	'p', 
	'eps_dist',
	'lmda_dist',
	'heterosked',
	'r2',
	'betaW_norm',
	'betaS_norm',
	'covmethod',
	'tau',
	'tauv',
	'stau',
	'lower_estimand_nox',
	'upper_estimand_nox',
	'lower_estimand',
	'upper_estimand',
	'lower_bound',
	'upper_bound',
	'runtime',
]

def compute_estimand(
	nrep_estimand,
	kappa,
	min_kappa,
	p,
	eps_dist,
	lmda_dist,
	heterosked,
	r2,
	betaW_norm, 
	betaS_norm,
	covmethod,
	tau,
	tauv,
	stau,
	t0,
	**args
):
	"""
	Computes partial identifiability estimands.
	"""
	# don't do unnecessary computation bc estimand
	# does not change with n
	if kappa != min_kappa:
		return []

	# parse args
	n = nrep_estimand
	dgp_args = [
		n, p, eps_dist, lmda_dist, heterosked, r2, 
		betaW_norm, betaS_norm, covmethod, tau, tauv, stau
	]


	# report
	msg = f"Computing estimand for p={p}, heterosked={heterosked}, "
	msg += f"eps_dist={eps_dist} at {elapsed(t0)}."
	print(msg)


	# Sample data
	data = gen_data.gen_lee_bound_data(
		sample_seed=1,
		n=n,
		p=p, 
		eps_dist=eps_dist,
		lmda_dist=lmda_dist,
		heterosked=heterosked,
		r2=r2,
		covmethod=covmethod,
		betaW_norm=betaW_norm, 
		betaS_norm=betaS_norm,
		tau=tau,
		stau=stau,
		tauv=tauv,
	)

	conditional_estimands = lee.compute_analytical_lee_bound(
		s0_probs=data['s0_probs'],
		s1_probs=data['s1_probs'],
		y0_dists=data['y0_dists'],
		y1_dists=data['y1_dists'],
	)
	estimands, bounds = utilities.compute_est_bounds(summands=conditional_estimands)

	# estimand without X
	bounds_nox = lee.lee_bound_no_covariates(
		W=data['W'], S=data['S'], Y=data['Y'],
	).tolist()


	runtime = time.time() - t0
	return dgp_args + bounds_nox + estimands.tolist() + bounds.tolist() + [runtime]

def single_seed_sim(
	seed,
	kappa,
	p,
	eps_dist,
	lmda_dist,
	heterosked,
	r2,
	betaW_norm, 
	betaS_norm,
	covmethod,
	tau,
	tauv,
	stau,
	t0,
	**args,
):
	n = int(kappa * p)
	dgp_args = [
		seed, n, p, eps_dist, lmda_dist, heterosked, r2, 
		betaW_norm, betaS_norm, covmethod, tau, tauv, stau
	]
	# report
	msg = f"At seed={seed}, kappa={kappa}, p={p}, heterosked={heterosked}, "
	msg += f"eps_dist={eps_dist} at {elapsed(t0)}."
	print(msg)


	# sample data
	data = gen_data.gen_lee_bound_data(
		sample_seed=seed,
		n=n,
		p=p, 
		eps_dist=eps_dist,
		lmda_dist=lmda_dist,
		heterosked=heterosked,
		r2=r2,
		covmethod=covmethod,
		betaW_norm=betaW_norm, 
		betaS_norm=betaS_norm,
		tau=tau,
		tauv=tauv,
		stau=stau,
	)

	# fit dual bounds
	alpha = args.get("alpha", 0.1)
	t0 = time.time()
	ldb = lee.LeeDualBounds(nvals=args.get("nvals"))
	ldb_args = dict(
	   Y=data['Y'],
	   S=data['S'],
	   W=data['W'],
	   X=data['X'],
	   pis=data['pis'],
	   verbose=args.get("verbose", False),
	)
	_ = ldb.compute_dual_bounds(
		nfolds=args.get("nfolds", DEFAULT_NFOLDS),
		**ldb_args,
	)
	runtime = time.time() - t0

	# oracle bounds
	t0 = time.time()
	ldb_oracle = lee.OracleLeeBounds(nvals=args.get("nvals"))
	_ = ldb_oracle.compute_oracle_bounds(
		s0_probs=data['s0_probs'],
		s1_probs=data['s1_probs'],
		y0_dists=data['y0_dists'],
		y1_dists=data['y1_dists'],
		**ldb_args
	)
	oracle_runtime = time.time() - t0

	# append to output
	output = []
	for summands, method, runtime in [
		(ldb.objvals - ldb.y0s0_cond_means, 'plugin', runtime),
		(ldb.ipw_summands, 'dual_crossfit_ipw', runtime),
		(ldb.aipw_summands, 'dual_crossfit_aipw', runtime),
		(ldb_oracle.ipw_summands, 'oracle_ipw', oracle_runtime),
		(ldb_oracle.aipw_summands, 'oracle_aipw', oracle_runtime),
	]:
		ests, cbs = utilities.compute_est_bounds(
			summands, alpha=alpha
		)
		output.append(
			dgp_args + [method] + ests.tolist() + cbs.tolist() + [runtime]
		)

	# bound without covariate
	t0 = time.time()
	lest_nox, uest_nox = lee.lee_bound_no_covariates(
		W=data['W'], S=data['S'], Y=data['Y'],
	)
	nocov_time = time.time() - t0
	output.append(
		dgp_args + ['nox'] + [lest_nox, uest_nox, np.nan, np.nan] + [nocov_time],
	)
	
	# return
	return output

def main(args):
	# Parse arguments
	args = parser.parse_args(args)
	reps = args.get('reps', [1])[0]
	seed_start = args.get("seed_start", [1])[0]
	num_processes = args.pop('num_processes', [5])[0]

	#### Key defaults go here
	# data-generating process
	args['p'] = args.get("p", [50])
	args['kappa'] = args.get("kappa", [2])
	args['eps_dist'] = args.get("eps_dist", ['gaussian'])
	args['lmda_dist'] = args.get("lmda_dist", ['constant'])
	args['heterosked'] = args.get("heterosked", ['none'])
	args['r2'] = args.get("r2", [0.5])
	args['betaS_norm'] = args.get('betas_norm', [1])
	args['betaW_norm'] = args.get('betaw_norm', [0])
	args['covmethod'] = args.get("covmethod", ['identity'])
	args['tau'] = args.get("tau", [0])
	args['stau'] = args.get("stau", [1])
	args['tauv'] = args.get("tauv", [1])

	# Some args can only have one value
	nrep_estimand = args.pop("nrep_estimand", [10000])[0]
	args['nvals'] = args.get("nvals", [50])
	if len(args['nvals']) > 1:
		raise ValueError(f"Can only provide one option for nvals, not {args['nvals']}")

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	args.pop("description")
	t0 = time.time()

	# Compute estimands
	estimands = utilities.apply_pool_factorial(
		func=compute_estimand, 
		num_processes=num_processes, 
		nrep_estimand=[nrep_estimand],
		min_kappa=[min(args['kappa'])],
		t0=[t0],
		**args,
	)
	estimands = [x for x in estimands if len(x) != 0]
	est_df = pd.DataFrame(estimands, columns=ESTIMAND_COLUMNS)
	est_df.to_csv(output_dir + "estimands.csv", index=False)
	print(est_df)

	# prepare to merge
	est_df = est_df.rename(
		columns={"lower_est":"lower_estimand", "upper_est":"upper_estimand"}
	).drop(
		["runtime", 'lower_bound', 'upper_bound', 'n'],
		axis='columns'
	)

	# Compute estimators
	outputs = utilities.apply_pool_factorial(
		func=single_seed_sim,
		seed=list(range(seed_start, reps+seed_start)), 
		num_processes=num_processes, 
		t0=[t0],
		**args,
	)
	out_df = []
	for output in outputs:
		out_df.extend(output)
	out_df = pd.DataFrame(out_df, columns=COLUMNS)

	# merge with estimands
	orig_n = len(out_df)
	merged_df = pd.merge(
		est_df, out_df,
		on=[
			'p', 'eps_dist', 'lmda_dist', 'heterosked',
			'r2', 'betaS_norm', 'betaW_norm',
			'covmethod', 'tau', 'tauv', 'stau',
		],
		how='inner',
	)
	if orig_n != len(merged_df):
		print("Merge failed, saving without merging")
		out_df.to_csv(output_dir + "results.csv", index=False)
	else:
		print("Merge succeeded, analyzing and saving.")

		# compute bias and SES for no_X plugin
		# this is an oracle estimator
		alpha = args.get("alpha", [0.1])[0] 
		scale = stats.norm.ppf(1-alpha/2)
		subset = merged_df.loc[merged_df['method'] == 'nox'].copy()
		subset['lower_diff'] = subset['lower_est'] - subset['lower_estimand_nox']
		subset['upper_diff'] = subset['upper_est'] - subset['upper_estimand_nox']
		ses = subset.groupby(GROUP_VALS)[[
			'lower_est', 'upper_est', 'lower_diff', 'upper_diff'
		]].agg(['mean', 'std']).reset_index()
		ses['lower_se'] = ses['lower_est']['std']
		ses['upper_se'] = ses['upper_est']['std']
		ses['lower_bias'] = ses['lower_diff']['mean']
		ses['upper_bias'] = ses['upper_diff']['mean']
		ses = ses[GROUP_VALS + ['lower_se', 'upper_se', 'lower_bias', 'upper_bias']]
		ses.columns = ses.columns.droplevel(1)
		subset = subset.drop(
			['lower_diff', 'upper_diff'], axis='columns'
		)
		subset = pd.merge(subset, ses, on=GROUP_VALS)
		subset['lower_bound'] = subset['lower_est'] - subset['lower_bias'] - scale * subset['lower_se']
		subset['upper_bound'] = subset['upper_est'] - subset['upper_bias'] + scale * subset['upper_se']
		subset = subset.drop(['lower_se', 'upper_se', 'lower_bias', 'upper_bias'], axis='columns')
		merged_df = pd.concat([merged_df.loc[merged_df['method'] != 'nox'], subset], axis='index')

		# coverage for methods which model X
		merged_df['lower_cov'] = merged_df['lower_bound'] <= merged_df['lower_estimand']
		merged_df['upper_cov'] = merged_df['upper_bound'] >= merged_df['upper_estimand']
		# coverage for methods which don't incorporate covariates
		flags = merged_df['method'] == 'nox'
		merged_df.loc[flags, 'lower_cov'] = merged_df.loc[flags, 'lower_bound'] <= merged_df.loc[flags, 'lower_estimand_nox']
		merged_df.loc[flags, 'upper_cov'] = merged_df.loc[flags, 'upper_bound'] >= merged_df.loc[flags, 'upper_estimand_nox']
		merged_df.to_csv(output_dir + "results.csv", index=False)

	# print output
	print(merged_df.groupby(GROUP_VALS)[
		'lower_cov', 'lower_est', 'lower_bound', 'lower_estimand', 
		'upper_cov', 'upper_est', 'upper_bound', 'upper_estimand',
		'runtime'
	].mean())


if __name__ == '__main__':
	main(sys.argv)