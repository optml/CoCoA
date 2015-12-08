/*
 * randomNumbersUtil.h
 *
 *  Created on: Jul 2, 2013
 *      Author: taki
 */

#ifndef LARGESCALEEXPERIMENT_H_
#define LARGESCALEEXPERIMENT_H_
#include "../solver/structures.h"
#include "../class/loss/losses.h"
#include "../helpers/gsl_random_helper.h"

namespace largeScaleExperiment {

template<typename L, typename D, typename LT>
void run_computation(ProblemData<L, D> &inst, double fvalOpt, int omp_threads,
		int N, int blockReduction, std::vector<gsl_rng *>& rs,
		ofstream& experimentLogFile, const int MAXIMUM_THREADS) {
	L n = inst.n;
	L m = inst.m;
	L sigma = inst.sigma;
	std::vector<D> h_Li(n, 0);
	Losses<L, D, LT>::compute_reciprocal_lipschitz_constants(inst, h_Li);
	omp_set_num_threads(omp_threads);
	randomNumberUtil::init_random_seeds(rs);
	inst.x.resize(n);
	for (L i = 0; i < n; i++)
		inst.x[i] = 0;
	std::vector<D> residuals(m);

	Losses<L, D, LT>::bulkIterations(inst, residuals);
	D fvalInit = Losses<L, D, LT>::compute_fast_objective(inst, residuals);

	double totalRunningTime = 0;
	double iterations = 0;
	L perPartIterations = n / blockReduction;
	double additional = perPartIterations / (0.0 + n);
	D fval = fvalInit;

// store initial objective value
	experimentLogFile << setprecision(16) << omp_threads << "," << n << "," << m
			<< "," << sigma << "," << totalRunningTime << "," << iterations
			<< "," << fval << endl;
	//iterate
	for (int totalIt = 0; totalIt < N; totalIt++) {

		double startTime = gettime_();
#pragma omp parallel for
		for (L it = 0; it < perPartIterations; it++) {
			// one step of the algorithm

			unsigned long int idx = gsl_rng_uniform_int(gsl_rng_r, n);
			Losses<L, D, LT>::do_single_iteration_parallel(inst, idx, residuals,
					inst.x, h_Li);
		}
		double endTime = gettime_();
		iterations += additional;
		totalRunningTime += endTime - startTime;
		// recompute residuals  - this step is not necessary but if accumulation of rounding errors occurs it is useful
		omp_set_num_threads(MAXIMUM_THREADS);
		if (totalIt % 3 == 0)
		{
			Losses<L, D, LT>::bulkIterations(inst, residuals);
		}
		fval = Losses<L, D, LT>::compute_fast_objective(inst, residuals);
		int nnz = 0;
#pragma omp parallel for reduction(+:nnz)
		for (L i = 0; i < n; i++)
			if (inst.x[i] != 0)
				nnz++;
		omp_set_num_threads(omp_threads);
		cout << omp_threads << "," << n << "," << m << "," << sigma << ","
				<< totalRunningTime << "," << iterations << "," << fval << ","
				<< fvalOpt << "," << nnz << "," << inst.oneZeroAccuracy << ","
				<< inst.primalObjective << "," << inst.dualObjective << ","
				<< inst.primalObjective - inst.dualObjective << endl;

		experimentLogFile << setprecision(16) << omp_threads << "," << n << ","
				<< m << "," << sigma << "," << totalRunningTime << ","
				<< iterations << "," << fval << "," << fvalOpt << "," << nnz
				<< "," << inst.oneZeroAccuracy << "," << inst.primalObjective
				<< "," << inst.dualObjective << ","
				<< inst.primalObjective - inst.dualObjective << endl;
	}
}

template<typename L, typename D, typename LT>
void run_experiment(ProblemData<L, D> &inst, int N, int PARTITION,std::vector<gsl_rng *>& rs,
		ofstream& experimentLogFile, const int MAXIMUM_THREADS, D fvalOpt = 0) {
	//-------------- set the number of threads which should be used.
	int TH[6] = { 24, 16, 8, 4, 2, 1 };
	for (int i = 0; i < 6; i++) {
		cout << "Running experiment with " << TH[i] << " threads" << endl;
		inst.sigma = 1 + (TH[i] - 1) * (inst.omega - 1) / (inst.n - 1);
		cout << setprecision(16) << "beta = " << inst.sigma << endl;
		largeScaleExperiment::run_computation<L, D, LT>(inst, fvalOpt, TH[i], N,
				PARTITION, rs, experimentLogFile, MAXIMUM_THREADS);
	}
}

}

#endif /* LARGESCALEEXPERIMENT_H_ */
