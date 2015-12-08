/*
 * ClusterSquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef CLUSTERSQUARELOSS_H_
#define CLUSTERSQUARELOSS_H_
#include "../../distributed/distributed_essentials.h"
#include "SquareLoss.h"
namespace Loss {
template<typename L, typename D>
class ClusterSquareLoss: public Loss::SquareLoss<L, D> {

public:

	ProblemData<L, D>* localInstance;
	std::vector<D>* localResiduals;
	ProblemData<L, D>* globalInstance;
	std::vector<D>* globalResiduals;
	std::vector<D>* residualsBuffer;
	D* residual_updatesPtr;
	mpi::communicator *world;
	ClusterSquareLoss() {
		this->localInstance = NULL;
		this->localResiduals = NULL;
		this->globalInstance = NULL;
		this->globalResiduals = NULL;
		this->residual_updatesPtr = NULL;
		this->world = NULL;
	}

	virtual inline void performOneCoordinateUpdate(L idx) {
		ProblemData<L, D>& instLocal = *this->localInstance;
		ProblemData<L, D>& instGlob = *this->globalInstance;

		D tmp = 0;
		tmp = compute_update(idx);
		parallel::atomic_add(instLocal.x[idx], tmp);
		for (L j = instLocal.A_csc_col_ptr[idx];
				j < instLocal.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(
					(*this->localResiduals)[instLocal.A_csc_row_idx[j]],
					tmp * instLocal.A_csc_values[j]);
		}
		for (L j = instGlob.A_csc_col_ptr[idx];
				j < instGlob.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(
					this->residual_updatesPtr[instGlob.A_csc_row_idx[j]],
					tmp * instGlob.A_csc_values[j]);
		}
	}

	void recomputeResiduals() {
		ProblemData<L, D>& instLocal = *this->localInstance;
		ProblemData<L, D>& instGlob = *this->globalInstance;
		// set r=-b+A*x on local data
		for (L i = 0; i < instLocal.m; i++)
			(*this->localResiduals)[i] = -instLocal.b[i];
		for (L i = 0; i < instLocal.n; i++) {
			for (L j = instLocal.A_csc_col_ptr[i];
					j < instLocal.A_csc_col_ptr[i + 1]; j++) {
				(*this->localResiduals)[instLocal.A_csc_row_idx[j]] +=
						instLocal.A_csc_values[j] * instLocal.x[i];
			}
		}
		// set r=A*x on global data
		for (L i = 0; i < instGlob.m; i++)
			(*this->residualsBuffer)[i] = 0;
		for (L i = 0; i < instGlob.n; i++) {
			for (L j = instGlob.A_csc_col_ptr[i];
					j < instGlob.A_csc_col_ptr[i + 1]; j++) {
				(*this->residualsBuffer)[instGlob.A_csc_row_idx[j]] +=
						instGlob.A_csc_values[j] * instLocal.x[i];
			}
		}

		// reduce all
		vall_reduce(*world, *this->residualsBuffer, *this->globalResiduals);
		for (L i = 0; i < instGlob.m; i++) {
			(*this->globalResiduals)[i] -= instGlob.b[i];
		}
	}

	virtual inline D compute_update(const L idx) {
		return 0;
	}
	virtual ~ClusterSquareLoss() {

	}
};

}
#endif /* CLUSTERSQUARELOSS_H_ */
