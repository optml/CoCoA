/*
 * SquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef MulticoreSVMDualLoss_H_
#define MulticoreSVMDualLoss_H_

#include "./SVMDualLoss.h"

namespace Loss {

template<typename L, typename D>
class MulticoreSVMDualLoss: public Loss::SVMDualLoss<L, D> {
public:

	ProblemData<L, D> instance;
	std::vector<D> residuals;

	D oneOverLambdaN;


	MulticoreSVMDualLoss(ProblemData<L, D>& _instance,
			std::vector<D>& _residuals) :
			instance(_instance), residuals(_residuals) {

	}

	void recomputeResiduals() {
		cblas_set_to_zero(residuals);
		const D delta = 1 / (instance.lambda * instance.n + 0.0);
		for (L sample = 0; sample < instance.n; sample++) {
			if (instance.x[sample] > 1)
				instance.x[sample] = 1;
			if (instance.x[sample] < 0)
				instance.x[sample] = 0;
			for (L tmp = instance.A_csr_row_ptr[sample];
					tmp < instance.A_csr_row_ptr[sample + 1]; tmp++) {
				residuals[instance.A_csr_col_idx[tmp]] += delta
						* instance.b[sample] * instance.A_csr_values[tmp]
						* instance.x[sample];
			}
		}
	}

	void computeReciprocalLipConstants() {
		instance.Li.resize(instance.n);
		for (L i = 0; i < instance.n; i++) {
			instance.Li[i] = 0;
			for (L j = instance.A_csr_row_ptr[i];
					j < instance.A_csr_row_ptr[i + 1]; j++) {
				instance.Li[i] += instance.A_csr_values[j]
						* instance.A_csr_values[j];
			}
			if (instance.Li[i] > 0)
				instance.Li[i] = 1 / (instance.sigma * instance.Li[i]); // Compute reciprocal Lipschitz Constants
		}

	}

	inline D computeObjectiveValue() {
		D resids = 0;
		D sumx = 0;
		D sumLoss = 0;
		resids = cblas_l2_norm(instance.m, &residuals[0], 1);
		resids = resids * resids;

		L good = 0;
		instance.dualObjective = 0;
		for (L j = 0; j < instance.n; j++) {
			D error = 0;
			for (L i = instance.A_csr_row_ptr[j];
					i < instance.A_csr_row_ptr[j + 1]; i++) {
				error += instance.A_csr_values[i]
						* residuals[instance.A_csr_col_idx[i]];
			}
			if (instance.b[j] * error > 0) {
				good++;
			}
			error = 1 - instance.b[j] * error;
			if (error < 0) {
				error = 0;
			}
			sumLoss += error;
			sumx += instance.x[j];
		}
		this->lastPrimalObjectiveValue = instance.lambda * 0.5 * resids
				+ (sumLoss) / (0.0 + instance.n);
		this->lastDualObjectiveValue = -instance.lambda * 0.5 * resids
				+ sumx / (0.0 + instance.n);
		this->lastZeroOneAccuracy = good / (0.0 + instance.n);

		this->lastTestZeroOneAccuracy = 0;
		if (instance.test_b.size() > 0) {
			good = 0;
			L all = 0;
			for (L j = 0; j < instance.A_test_csr_row_ptr.size() - 1; j++) {
				D error = 0;
				for (L i = instance.A_test_csr_row_ptr[j];
						i < instance.A_test_csr_row_ptr[j + 1]; i++) {
					error += instance.A_test_csr_values[i]
							* residuals[instance.A_test_csr_col_idx[i]];
				}
				if (instance.test_b[j] * error > 0) {
					good++;
				}
				all++;
			}
			this->lastTestZeroOneAccuracy = good / (all + 0.0);
		}

		this->lastComputedObjectiveValue = this->lastPrimalObjectiveValue
				- this->lastDualObjectiveValue;
		return this->lastComputedObjectiveValue;
	}

	inline void performOneCoordinateUpdate(L idx) {
		D tmp = 0;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {
			tmp += instance.A_csr_values[i]
					* residuals[instance.A_csr_col_idx[i]];
		}
		tmp = (1 - tmp * instance.b[idx]) * instance.lambda * instance.n
				* instance.Li[idx];

		if (tmp < -instance.x[idx]) {
			tmp = -instance.x[idx];
		} else if (tmp > 1 - instance.x[idx]) {
			tmp = 1 - instance.x[idx];
		}
		parallel::atomic_add(instance.x[idx], tmp);
		const D delta = 1 / (instance.lambda * instance.n);
		for (L j = instance.A_csr_row_ptr[idx];
				j < instance.A_csr_row_ptr[idx + 1]; j++) {
			parallel::atomic_add(residuals[instance.A_csr_col_idx[j]],
					tmp * instance.A_csr_values[j] * instance.b[idx] * delta);
		}
	}

	virtual ~MulticoreSVMDualLoss() {

	}
};

}
#endif /* SQUARELOSS_H_ */
