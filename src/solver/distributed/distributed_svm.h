/*
 * distributed_svm.h
 *
 *  Created on: Feb 15, 2012
 *      Author: taki
 */

#ifndef DISTRIBUTED_SVM_H_
#define DISTRIBUTED_SVM_H_

#include "distributed_include.h"


#include "distributed_structures.h"

using namespace std;
namespace mpi = boost::mpi;

template<typename D, typename L>
double compute_distributed_prediction_accuracy_for_svm(mpi::environment &env, mpi::communicator &world,
		ProblemData<L, D> &test_data, ProblemData<L, D> &train_data) {
	L total_samples = test_data.m;
	L total_correct_predictions = 0;
	L max_x_length = train_data.x.size();
	for (L sample = 0; sample < total_samples; sample++) {
		D prediction = 0;
		for (L j = test_data.A_csr_row_ptr[sample]; j < test_data.A_csr_row_ptr[sample + 1]; j++) {
			if (test_data.A_csr_col_idx[j] < max_x_length) {
				prediction += test_data.b[sample] * test_data.A_csr_values[j]
						* train_data.x[test_data.A_csr_col_idx[j]];
			}
		}
		if (prediction > 0) {
			total_correct_predictions++;
		}
	}
	L total_samples_from_all_dataset = 0;
	L total_correct_predictions_from_all_dataset = 0;
	reduce(world, total_samples, total_samples_from_all_dataset, std::plus<L>(), 0);
	reduce(world, total_correct_predictions, total_correct_predictions_from_all_dataset, std::plus<L>(), 0);
	if (world.rank() == 0) {
		return (double) total_correct_predictions_from_all_dataset / total_samples_from_all_dataset;
	}
	return -1;
}

template<typename D, typename L>
double compute_distributed_prediction_accuracy_for_svm_for_part(mpi::environment &env,
		mpi::communicator &world, ProblemData<L, D> &test_data) {
	L total_samples = test_data.m;
	std::vector < D > myPartialPrediction(total_samples );
	std::vector < D> buffer_myPartialPrediction(total_samples );
	DistributedLosses<L, D, square_hinge_loss_traits>::bulkIterations_for_my_part_data(test_data,
			myPartialPrediction, buffer_myPartialPrediction, world);
	if (world.rank() == 0) {
		L total_correct_predictions = 0;
		for (L sample = 0; sample < total_samples; sample++) {
			D prediction = myPartialPrediction[sample];
			if (prediction < 0) { //usually there is > 0  but our function bulkIterations_for_my_part_data returns data mupliplied by "-1"
				total_correct_predictions++;
			}
		}
		return (double) total_correct_predictions / (0.0 + total_samples);
	}
	return -1;
}

#endif /* DISTRIBUTED_SVM_H_ */
