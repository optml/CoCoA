/*
 * SquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef APPROXClusterSVMDualLoss_H_
#define APPROXClusterSVMDualLoss_H_

#include "ClusterSVMDualLoss.h"

namespace Loss {

template<typename L, typename D>
class ApproxClusterSVMDualLoss: public Loss::ClusterSVMDualLoss<L, D> {
public:

	ApproxClusterSVMDualLoss(mpi::communicator &_world,
			ProblemData<L, D>& _localInstance, std::vector<D>& _localResiduals,
			ProblemData<L, D>& _globalInstance,
			std::vector<D>& _globalResiduals, std::vector<D>& _residualsBuffer,
			D* _residual_updatesPtr) :
			Loss::ClusterSVMDualLoss<L, D>(_world, _localInstance,
					_localResiduals, _globalInstance, _globalResiduals,
					_residualsBuffer, _residual_updatesPtr) {




	}

	virtual void recomputeResiduals() {
		cblas_set_to_zero(this->residualsBuffer);
		const D delta = 1
				/ (this->localInstance.lambda * this->localInstance.total_n
						+ 0.0);
		for (L sample = 0; sample < this->globalInstance.n; sample++) {
//			if (localInstance.x[sample] > 1)
//				localInstance.x[sample] = 1;
//			if (localInstance.x[sample] < 0)
//				localInstance.x[sample] = 0;

			for (L tmp = this->globalInstance.A_csr_row_ptr[sample];
					tmp < this->globalInstance.A_csr_row_ptr[sample + 1];
					tmp++) {
				this->residualsBuffer[this->globalInstance.A_csr_col_idx[tmp]] +=
						delta * this->globalInstance.b[sample]
								* this->globalInstance.A_csr_values[tmp]
								* this->localInstance.z[sample];
				this->residualsBuffer[this->globalInstance.m
						+ this->globalInstance.A_csr_col_idx[tmp]] += delta
						* this->globalInstance.b[sample]
						* this->globalInstance.A_csr_values[tmp]
						* this->localInstance.u[sample];

			}
		}
		vall_reduce(this->world, this->residualsBuffer, this->globalResiduals);
	}

	virtual inline D computeObjectiveValue() {
		D resids = 0;
		D sumLoss = 0;
		D sumX = 0;
		resids = cblas_l2_norm(this->globalInstance.m,
				&this->globalResiduals[0], 1);
		resids = resids * resids;
		L good = 0;

		D thetaSq = this->localInstance.prevTheta
				* this->localInstance.prevTheta;

		for (L j = 0; j < this->localInstance.n; j++) {
			D error = 0;
			for (L i = this->globalInstance.A_csr_row_ptr[j];
					i < this->globalInstance.A_csr_row_ptr[j + 1]; i++) {
				error +=
						this->globalInstance.A_csr_values[i]
								* (this->globalResiduals[this->globalInstance.A_csr_col_idx[i]]
										+ thetaSq
												* this->globalResiduals[this->globalInstance.A_csr_col_idx[i]
														+ this->globalInstance.m]);
			}
			if (this->globalInstance.b[j] * error > 0) {
				good++;
			}
			error = 1 - this->globalInstance.b[j] * error;
			if (error < 0) {
				error = 0;
			}
			sumLoss += error;
			sumX += this->localInstance.z[j]
					+ thetaSq * this->localInstance.u[j];
		}
		D sumxLossOut = 0;
		reduce(this->world, sumLoss, sumxLossOut, std::plus<D>(), 0);
		D sumXOut = 0;
		reduce(this->world, sumX, sumXOut, std::plus<D>(), 0);
		L sumxGood = 0;
		reduce(this->world, good, sumxGood, std::plus<L>(), 0);

		this->lastZeroOneAccuracy = sumxGood
				/ (0.0 + this->globalInstance.total_n);
		this->lastPrimalObjectiveValue = this->globalInstance.lambda * 0.5
				* resids + sumxLossOut / (0.0 + this->globalInstance.total_n);
		this->lastDualObjectiveValue = -this->globalInstance.lambda * 0.5
				* resids + sumXOut / (0.0 + this->globalInstance.total_n);
		this->lastComputedObjectiveValue = this->lastPrimalObjectiveValue
				- this->lastDualObjectiveValue;
		return this->lastComputedObjectiveValue;
	}

	virtual inline void performOneCoordinateUpdate(L idx) {
		D gamma = this->localInstance.total_n * this->localInstance.theta / this->localInstance.Li[idx]
				/ (0.0 + this->localInstance.total_tau);
		D tmp = 0;
		D thetaSq = this->localInstance.theta * this->localInstance.theta;
		for (L i = this->globalInstance.A_csr_row_ptr[idx];
				i < this->globalInstance.A_csr_row_ptr[idx + 1]; i++) {
			tmp +=
					this->globalInstance.A_csr_values[i]
							* (this->globalResiduals[this->globalInstance.A_csr_col_idx[i]]
									+ thetaSq
											* this->globalResiduals[this->globalInstance.A_csr_col_idx[i]
													+ this->globalInstance.m]);
		}
		tmp = (1 - tmp * this->globalInstance.b[idx])
				* this->globalInstance.lambda * this->globalInstance.n / gamma;

		if (tmp < -this->localInstance.z[idx]) {
			tmp = -this->localInstance.z[idx];
		} else if (tmp > 1 - this->localInstance.z[idx]) {
			tmp = 1 - this->localInstance.z[idx];
		}

		parallel::atomic_add(this->localInstance.z[idx], tmp);
		D du = -(1
				- this->localInstance.total_n / (0.0 + this->localInstance.total_tau)
						* this->localInstance.theta) / thetaSq;


		parallel::atomic_add(this->localInstance.u[idx], du * tmp);

		const D delta = 1
				/ (this->globalInstance.lambda * this->globalInstance.total_n);
		for (L j = this->globalInstance.A_csr_row_ptr[idx];
				j < this->globalInstance.A_csr_row_ptr[idx + 1]; j++) {
			parallel::atomic_add(
					this->residual_updatesPtr[this->globalInstance.A_csr_col_idx[j]],
					tmp * this->globalInstance.A_csr_values[j]
							* this->globalInstance.b[idx] * delta);

			parallel::atomic_add(
					this->residual_updatesPtr[this->globalInstance.m
							+ this->globalInstance.A_csr_col_idx[j]],
					tmp * du * this->globalInstance.A_csr_values[j]
							* this->globalInstance.b[idx] * delta);

		}
	}
	virtual ~ApproxClusterSVMDualLoss() {

	}
};

}
#endif /* SQUARELOSS_H_ */
