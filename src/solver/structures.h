/*
 * This file contains an functions for experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */

/*
 * Optimization and output structures,
 *
 * OptimizationSettings structure is used to configure solvers, how much can solver run,
 */

#ifndef OPTIMIZATION_STRUCTURES_H_
#define OPTIMIZATION_STRUCTURES_H_

#include <ios>
#include <cstdlib>
#include <unistd.h>

#include "sys/types.h"
#include "sys/sysinfo.h"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <string>

template<typename L, typename D>
class ProblemData {
public:
	std::vector<D> A_csc_values;
	std::vector<L> A_csc_row_idx;
	std::vector<L> A_csc_col_ptr;

	std::vector<D> A_csr_values;
	std::vector<L> A_csr_col_idx;
	std::vector<L> A_csr_row_ptr;

	std::vector<D> A_test_csr_values;
	std::vector<L> A_test_csr_col_idx;
	std::vector<L> A_test_csr_row_ptr;

	std::vector<D> test_b;

	std::vector<D> Li;
	std::vector<D> vi; // ESO stepsize

	std::string experimentName;

	D oneOverLambdaN;

	D penalty;

	std::vector<D> b;
#ifdef NEWATOMICS
	std::vector< atomic_float<D, L> > x;
#else
	std::vector<D> x;

	std::vector<D> z;
	std::vector<D> u;
	D prevTheta;
	D theta;
	D tau;

#endif

	L m;
	L n;
	L total_n;
	L total_tau;
	D lambda;
	D omega;
	D sigma;

	D omegaMin;
	D omegaAvg;

	D oneZeroAccuracy;
	D dualObjective;
	D primalObjective;
	D* A_csr_values_raw;
	L* A_csr_col_idx_raw;
	L* A_csr_row_ptr_raw;
	D* b_raw;
	D* x_raw;
	void init_pointers_for_CSR() {
		A_csr_values_raw = &(A_csr_values)[0];
		A_csr_col_idx_raw = &(A_csr_col_idx)[0];
		A_csr_row_ptr_raw = &(A_csr_row_ptr)[0];
		b_raw = &(b)[0];
		x_raw = &(x)[0];
	}

};

template<typename L, typename D>
class problem_mc_data {
public:
	std::vector<D> A_coo_values;
	std::vector<L> A_coo_row_idx;
	std::vector<L> A_coo_col_idx;
	std::vector<short int> A_coo_operator; // 0 is equality  -1 or +1 is inequality constraint
	D mu;
	L m;
	L n;
	L rank;

	std::vector<D> R_mat;
	std::vector<D> L_mat;
};

#endif /* OPTIMIZATION_STRUCTURES_H_ */

