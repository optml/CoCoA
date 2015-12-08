/*
 * SquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef ClusterSVMDualLoss_H_
#define ClusterSVMDualLoss_H_

#include "./SVMDualLoss.h"

namespace Loss {

template<typename L, typename D>
class ClusterSVMDualLoss: public Loss::SVMDualLoss<L, D> {
public:

	ProblemData<L, D>& localInstance;
	std::vector<D>& localResiduals;

	ProblemData<L, D>& globalInstance;
	std::vector<D> &globalResiduals;

	mpi::communicator &world;

	std::vector<D>& residualsBuffer;
	D* residual_updatesPtr;
	ClusterSVMDualLoss(mpi::communicator &_world,
			ProblemData<L, D>& _localInstance, std::vector<D>& _localResiduals,
			ProblemData<L, D>& _globalInstance,
			std::vector<D>& _globalResiduals, std::vector<D>& _residualsBuffer,
			D* _residual_updatesPtr) :
			localInstance(_localInstance), localResiduals(_localResiduals), globalInstance(
					_globalInstance), globalResiduals(_globalResiduals), world(
					_world), residualsBuffer(_residualsBuffer) {
		residual_updatesPtr = _residual_updatesPtr;
		vall_reduce(world, &localInstance.n, &globalInstance.total_n,1);
		localInstance.total_n = globalInstance.total_n;
	}

	virtual void recomputeResiduals() {
		cblas_set_to_zero(residualsBuffer);
		const D delta = 1
				/ (localInstance.lambda * localInstance.total_n + 0.0);
		for (L sample = 0; sample < globalInstance.n; sample++) {
			if (localInstance.x[sample] > 1)
				localInstance.x[sample] = 1;
			if (localInstance.x[sample] < 0)
				localInstance.x[sample] = 0;

			for (L tmp = globalInstance.A_csr_row_ptr[sample];
					tmp < globalInstance.A_csr_row_ptr[sample + 1]; tmp++) {
				residualsBuffer[globalInstance.A_csr_col_idx[tmp]] += delta
						* globalInstance.b[sample]
						* globalInstance.A_csr_values[tmp]
						* localInstance.x[sample];
			}
		}
		vall_reduce(world, residualsBuffer, globalResiduals);
	}

	virtual void computeReciprocalLipConstants() {
		localInstance.Li.resize(globalInstance.n);
		for (L i = 0; i < globalInstance.n; i++) {
			localInstance.Li[i] = 0;
			for (L j = globalInstance.A_csr_row_ptr[i];
					j < globalInstance.A_csr_row_ptr[i + 1]; j++) {
				localInstance.Li[i] += globalInstance.A_csr_values[j]
						* globalInstance.A_csr_values[j];
			}
			if (localInstance.Li[i] > 0) {
				localInstance.Li[i] = 1
						/ (localInstance.sigma * localInstance.Li[i]); // Compute reciprocal Lipschitz Constants
			}
		}
	}

	virtual inline D computeObjectiveValue() {
		D resids = 0;
		D sumLoss = 0;
		D sumX = 0;
		resids = cblas_l2_norm(globalInstance.m, &globalResiduals[0], 1);
		resids = resids * resids;
		L good = 0;
		for (L j = 0; j < localInstance.n; j++) {
			D error = 0;
			for (L i = globalInstance.A_csr_row_ptr[j];
					i < globalInstance.A_csr_row_ptr[j + 1]; i++) {
				error += globalInstance.A_csr_values[i]
						* globalResiduals[globalInstance.A_csr_col_idx[i]];
			}
			if (globalInstance.b[j] * error > 0) {
				good++;
			}
			error = 1 - globalInstance.b[j] * error;
			if (error < 0) {
				error = 0;
			}
			sumLoss += error;
			sumX += localInstance.x[j];
		}
		D sumxLossOut = 0;
		reduce(world, sumLoss, sumxLossOut, std::plus<D>(), 0);
		D sumXOut = 0;
		reduce(world, sumX, sumXOut, std::plus<D>(), 0);
		L sumxGood = 0;
		reduce(world, good, sumxGood, std::plus<L>(), 0);

		this->lastZeroOneAccuracy = sumxGood / (0.0 + globalInstance.total_n);
		this->lastPrimalObjectiveValue = globalInstance.lambda * 0.5 * resids
				+ sumxLossOut / (0.0 + globalInstance.total_n);
		this->lastDualObjectiveValue = -globalInstance.lambda * 0.5 * resids
				+ sumXOut / (0.0 + globalInstance.total_n);
		this->lastComputedObjectiveValue = this->lastPrimalObjectiveValue
				- this->lastDualObjectiveValue;
		return this->lastComputedObjectiveValue;
	}

	virtual inline void performOneCoordinateUpdate(L idx) {
		D tmp = 0;
		for (L i = globalInstance.A_csr_row_ptr[idx];
				i < globalInstance.A_csr_row_ptr[idx + 1]; i++) {
			tmp += globalInstance.A_csr_values[i]
					* (globalResiduals[globalInstance.A_csr_col_idx[i]]);
		}
		tmp = (1 - tmp * globalInstance.b[idx]) * globalInstance.lambda
				* globalInstance.n * localInstance.Li[idx];

		if (tmp < -localInstance.x[idx]) {
			tmp = -localInstance.x[idx];
		} else if (tmp > 1 - localInstance.x[idx]) {
			tmp = 1 - localInstance.x[idx];
		}
		parallel::atomic_add(localInstance.x[idx], tmp);
		const D delta = 1 / (globalInstance.lambda * globalInstance.total_n);
		for (L j = globalInstance.A_csr_row_ptr[idx];
				j < globalInstance.A_csr_row_ptr[idx + 1]; j++) {
			parallel::atomic_add(
					this->residual_updatesPtr[globalInstance.A_csr_col_idx[j]],
					tmp * globalInstance.A_csr_values[j] * globalInstance.b[idx]
							* delta);
		}
	}

	virtual ~ClusterSVMDualLoss() {

	}
};

}
#endif /* SQUARELOSS_H_ */
