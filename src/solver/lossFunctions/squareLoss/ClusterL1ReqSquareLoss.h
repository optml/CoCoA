/*
 * ClusterL1ReqSquadeLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef CLUSTERL1REQSQUADELOSS_H_
#define CLUSTERL1REQSQUADELOSS_H_

#include "ClusterSquareLoss.h"
namespace Loss {
template<typename L, typename D>
class ClusterL1ReqSquareLoss: public Loss::ClusterSquareLoss<L, D> {
public:
	ClusterL1ReqSquareLoss(mpi::communicator *_world,
			ProblemData<L, D>* _localInstance, std::vector<D>* _localResiduals,
			ProblemData<L, D>* _globalInstance,
			std::vector<D>* _globalResiduals, std::vector<D>* _residualsBuffer,
			D* _residual_updatesPtr) {
		this->localInstance = _localInstance;
		this->localResiduals = _localResiduals;
		this->globalInstance = _globalInstance;
		this->globalResiduals = _globalResiduals;
		this->residualsBuffer = _residualsBuffer;
		this->residual_updatesPtr = _residual_updatesPtr;
		this->world = _world;

	}

	virtual ~ClusterL1ReqSquareLoss() {

	}

	inline D compute_update(const L idx) {
		D tmp = 0; //compute partial derivative f_idx'(x)
		for (unsigned int j = this->localInstance->A_csc_col_ptr[idx];
				j < this->localInstance->A_csc_col_ptr[idx + 1]; j++) {
			tmp +=
					this->localInstance->A_csc_values[j]
							* (*this->localResiduals)[this->localInstance->A_csc_row_idx[j]];
		}

		for (unsigned int j = this->globalInstance->A_csc_col_ptr[idx];
				j < this->globalInstance->A_csc_col_ptr[idx + 1]; j++) {
			tmp +=
					this->globalInstance->A_csc_values[j]
							* (*this->globalResiduals)[this->globalInstance->A_csc_row_idx[j]];
		}

		tmp = compute_soft_treshold(
				this->localInstance->Li[idx] * this->localInstance->lambda,
				this->localInstance->x[idx]
						- this->localInstance->Li[idx] * tmp)
				- this->localInstance->x[idx];
		return tmp;
	}

	D computeObjectiveValue() {
		ProblemData<L, D>& instLocal = *this->localInstance;
		ProblemData<L, D>& instGlob = *this->globalInstance;
		D localRedidsSq = cblas_l2_norm(instLocal.m,
				&(*this->localResiduals)[0], 1);
		localRedidsSq = localRedidsSq * localRedidsSq;

		D globalRedidsSq = cblas_l2_norm(instGlob.m,
				&(*this->globalResiduals)[0], 1);
		globalRedidsSq = globalRedidsSq * globalRedidsSq;

		D localRedidsSqOut = 0;
		D localL1Norm = cblas_l1_norm(instLocal.n, &instLocal.x[0], 1);
		D l1NormOut = 0;
		reduce(*this->world, localRedidsSq, localRedidsSqOut, std::plus<D>(),
				0);
		reduce(*this->world, localL1Norm, l1NormOut, std::plus<D>(), 0);
		this->lastComputedObjectiveValue = 0.5
				* (globalRedidsSq + localRedidsSqOut)
				+ instLocal.lambda * l1NormOut;
		return this->lastComputedObjectiveValue;
	}

	void computeReciprocalLipConstants() {
		ProblemData<L, D>& inst = *this->localInstance;
		ProblemData<L, D>& instGlob = *this->globalInstance;
		cout << "Allocate " << inst.n << " pole ";
		inst.Li.resize(inst.n);
		cout << "Allocate " << inst.n << " done ";
		cout << "XX "<<inst.A_csc_col_ptr.size()<<" "<<instGlob.A_csc_col_ptr.size()<<" "<<inst.A_csc_values.size()<<" "<< instGlob.A_csc_values.size()<<endl;
		for (L col = 0; col < inst.n; col++) {
			inst.Li[col] = 0;
			for (L i = inst.A_csc_col_ptr[col]; i < inst.A_csc_col_ptr[col + 1];
					i++) {
				inst.Li[col] += inst.A_csc_values[i] * inst.A_csc_values[i];
			}
			for (L i = instGlob.A_csc_col_ptr[col];
					i < instGlob.A_csc_col_ptr[col + 1]; i++) {
				inst.Li[col] += instGlob.A_csc_values[i]
						* instGlob.A_csc_values[i];
			}
			if (inst.Li[col] > 0)
				inst.Li[col] = 1 / inst.Li[col] / inst.sigma;
		}
	}

};
}

#endif /* CLUSTERL1REQSQUADELOSS_H_ */
