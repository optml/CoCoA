/*
 * MulticoreSquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef MulticoreL2SquareRegLogisticLoss_H_
#define MulticoreL2SquareRegLogisticLoss_H_
#include "MulticoreLogisticLoss.h"
#include "../../structures.h"
#include "../../treshhold_functions.h"
#include "../../../parallel/parallel_essentials_posix.h"
namespace Loss {

template<typename L, typename D>
class MulticoreL2SquareRegLogisticLoss: public Loss::MulticoreLogisticLoss<L, D> {
public:

	MulticoreL2SquareRegLogisticLoss(ProblemData<L, D>* _instance,
			std::vector<D>* _residuals) {
		this->inst = _instance;
		this->residuals = _residuals;

	}
	virtual ~MulticoreL2SquareRegLogisticLoss() {

	}

	void computeReciprocalLipConstants() {
		ProblemData<L, D>& instLocal = *this->inst;
		instLocal.Li.resize(instLocal.n);
		for (L i = 0; i < instLocal.n; i++) {
			instLocal.Li[i] = instLocal.lambda;
			for (L j = instLocal.A_csc_col_ptr[i];
					j < instLocal.A_csc_col_ptr[i + 1]; j++) {
				instLocal.Li[i] += 0.25 * instLocal.A_csc_values[j]
						* instLocal.A_csc_values[j];
			}
			instLocal.Li[i] = instLocal.Li[i] * instLocal.sigma;
			if (instLocal.Li[i] > 0)
				instLocal.Li[i] = 1 / instLocal.Li[i];
		}
	}

	inline D computeObjectiveValue() {
		D firstPart = 0;
		for (L i = 0; i < (*this->residuals).size(); i++) {
			firstPart += log(1 + exp((*this->residuals)[i]));
		}

		D l2Norm = cblas_l2_norm((*this->inst).x.size(), &(*this->inst).x[0],
				1);

		L good = 0;
		L all = 0;
		for (L i = 0; i < (*this->residuals).size(); i++) {
			all++;
			if ((*this->residuals)[i] < 0) {
				good++;
			}
		}
		this->trainAccuracy = good / (all + 0.0);

		good = 0;
		all = 0;
		ProblemData<L, D>& instLocal = *this->inst;
		for (L i = 0; i < instLocal.A_test_csr_row_ptr.size() - 1; i++) {
			all++;
			D tmp = 0;
			for (L j = instLocal.A_test_csr_row_ptr[i];
					j < instLocal.A_test_csr_row_ptr[i + 1]; j++) {
				tmp += instLocal.A_test_csr_values[j]
						* instLocal.x[instLocal.A_csr_col_idx[j]];
			}
			if (tmp * instLocal.test_b[i] > 0) {
				good++;
			}

		}
		this->testAccuracy = good / (all + 0.0);

		this->lastComputedObjectiveValue = firstPart
				+ (this->inst->lambda) * l2Norm * l2Norm * 0.5;
		return this->lastComputedObjectiveValue;
	}

	inline D compute_update(const ProblemData<L, D> &inst,
			const std::vector<D> &residuals, const L idx) {
		D tmp = 0; //compute partial derivative f_idx'(x)
		for (unsigned int j = inst.A_csc_col_ptr[idx];
				j < inst.A_csc_col_ptr[idx + 1]; j++) {
			D tt = exp(residuals[inst.A_csc_row_idx[j]]);
			tmp += -inst.b[inst.A_csc_row_idx[j]] * inst.A_csc_values[j] * tt
					/ (1 + tt);
		}
		tmp += this->inst->lambda * inst.x[idx];
		tmp = -inst.Li[idx] * tmp;
		return tmp;
	}

	inline void performOneCoordinateUpdate(L idx) {
		D tmp = 0;
		tmp = compute_update((*this->inst), (*this->residuals), idx);
		tmp = tmp / 2;
		parallel::atomic_add(this->inst->x[idx], tmp);

		for (L j = this->inst->A_csc_col_ptr[idx];
				j < this->inst->A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(
					(*this->residuals)[this->inst->A_csc_row_idx[j]],
					-this->inst->b[this->inst->A_csc_row_idx[j]] * tmp
							* this->inst->A_csc_values[j]);
		}
	}

};
}
#endif /* MulticoreL2SquareRegSquareLoss */
