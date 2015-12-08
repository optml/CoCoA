/*
 * MulticoreSquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef MulticoreL2SquareRegSquareLoss_H_
#define MulticoreL2SquareRegSquareLoss_H_
#include "MulticoreSquareLoss.h"
#include "../../structures.h"
#include "../../treshhold_functions.h"
#include "../../../parallel/parallel_essentials_posix.h"
namespace Loss {

template<typename L, typename D>
class MulticoreL2SquareRegSquareLoss: public Loss::MulticoreSquareLoss<L, D> {
public:

	MulticoreL2SquareRegSquareLoss(ProblemData<L, D>* _instance,
			std::vector<D>* _residuals) {
		this->inst = _instance;
		this->residuals = _residuals;

	}
	virtual ~MulticoreL2SquareRegSquareLoss() {

	}

	void computeReciprocalLipConstants() {
		ProblemData<L, D>& instLocal = *this->inst;
		instLocal.Li.resize(instLocal.n);
		for (L i = 0; i < instLocal.n; i++) {
			instLocal.Li[i] = instLocal.lambda;
			for (L j = instLocal.A_csc_col_ptr[i];
					j < instLocal.A_csc_col_ptr[i + 1]; j++) {
				instLocal.Li[i] += instLocal.A_csc_values[j]
						* instLocal.A_csc_values[j];
			}
			if (instLocal.Li[i] > 0)
				instLocal.Li[i] = 1 / instLocal.Li[i];
		}
	}

	inline D computeObjectiveValue() {
		D l2ResNorm = cblas_l2_norm((*this->residuals).size(),
				&(*this->residuals)[0], 1);
		D l2Norm = cblas_l2_norm((*this->inst).x.size(), &(*this->inst).x[0],
				1);

		this->lastComputedObjectiveValue = 0.5 * l2ResNorm * l2ResNorm
				+ (this->inst->lambda) * l2Norm * l2Norm * 0.5;
		return this->lastComputedObjectiveValue;
	}

	inline D compute_update(const ProblemData<L, D> &inst,
			const std::vector<D> &residuals, const L idx) {
		D tmp = 0; //compute partial derivative f_idx'(x)
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			tmp += inst.A_csc_values[j] * residuals[inst.A_csc_row_idx[j]];
		}
		tmp += this->inst->lambda * inst.x[idx];
		tmp = -inst.Li[idx] * tmp;
		return tmp;
	}

	inline void performOneCoordinateUpdate(L idx) {
		D tmp = 0;
		tmp = compute_update((*this->inst), (*this->residuals), idx);
		parallel::atomic_add(this->inst->x[idx], tmp);
		for (L j = this->inst->A_csc_col_ptr[idx];
				j < this->inst->A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(
					(*this->residuals)[this->inst->A_csc_row_idx[j]],
					tmp * this->inst->A_csc_values[j]);
		}
	}

};
}
#endif /* MulticoreL2SquareRegSquareLoss */
