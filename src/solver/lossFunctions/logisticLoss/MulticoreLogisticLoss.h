/*
 * MulticoreSquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef MULTICORELogisticLOSS_H_
#define MULTICORELogisticLOSS_H_
#include "LogisticLoss.h"
namespace Loss {
template<typename L, typename D>
class MulticoreLogisticLoss: public Loss::LogisticLoss<L, D> {
public:
	virtual ~MulticoreLogisticLoss() {

	}

	ProblemData<L, D>* inst;
	std::vector<D>* residuals;

	MulticoreLogisticLoss() {
		this->inst = NULL;
		this->residuals = NULL;
	}

	virtual void computeReciprocalLipConstants() {
//		ProblemData<L, D>& instLocal = *this->inst;
//		instLocal.Li.resize(instLocal.n);
//		for (L i = 0; i < instLocal.n; i++) {
//			instLocal.Li[i] = 0;
//			for (L j = instLocal.A_csc_col_ptr[i];
//					j < instLocal.A_csc_col_ptr[i + 1]; j++) {
//				instLocal.Li[i] += instLocal.A_csc_values[j]
//						* instLocal.A_csc_values[j];
//			}
//			if (instLocal.Li[i] > 0)
//				instLocal.Li[i] = 1 / instLocal.Li[i];
//		}
	}

	virtual void recomputeResiduals() {

		ProblemData<L, D>& instLocal = *this->inst;
		for (L i = 0; i < instLocal.m; i++)
			(*this->residuals)[i] = 0;
		for (L i = 0; i < instLocal.n; i++) {
			for (L j = instLocal.A_csc_col_ptr[i];
					j < instLocal.A_csc_col_ptr[i + 1]; j++) {
				(*residuals)[instLocal.A_csc_row_idx[j]] +=
						-instLocal.b[instLocal.A_csc_row_idx[j]]*
						instLocal.A_csc_values[j] * instLocal.x[i];
			}
		}
	}

};
}

#endif /* MULTICORESQUARELOSS_H_ */
