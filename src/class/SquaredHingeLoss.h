/*
 * SquaredHingeLoss.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef SQUAREDHINGELOSS_H_
#define SQUAREDHINGELOSS_H_
#include "LossFunction.h"

template<typename L, typename D>
class SquaredHingeLoss : public LossFunction<L, D>{
public:
	SquaredHingeLoss(){

	}



	virtual void computeObjectiveValue(ProblemData<L, D> & instance,
					mpi::communicator & world, std::vector<D> & w, double &finalDualError,
					double &finalPrimalError){

		D localError = 0;
			for (unsigned int i = 0; i < instance.n; i++) {
				D tmp = instance.x[i] * instance.x[i] / 4 - instance.x[i];
				localError += tmp;
			}

			D localSquaredHingeLoss = 0;
			for (unsigned int idx = 0; idx < instance.n; idx++) {
				D dotProduct = 0;
				for (L i = instance.A_csr_row_ptr[idx];
						i < instance.A_csr_row_ptr[idx + 1]; i++) {
					dotProduct += (w[instance.A_csr_col_idx[i]])
							* instance.A_csr_values[i];
				}
				D tmp = 1 - instance.b[idx] * dotProduct;

				if (tmp > 0) {
					localSquaredHingeLoss += tmp * tmp;
				}
			}
			finalPrimalError = 0;
			vall_reduce(world, &localSquaredHingeLoss, &finalPrimalError, 1);

			finalDualError = 0;
			vall_reduce(world, &localError, &finalDualError, 1);

			D tmp2 = cblas_l2_norm(w.size(), &w[0], 1);
			finalDualError = 1 / (0.0 + instance.total_n) * finalDualError
					+ 0.5 * instance.lambda * tmp2 * tmp2;
		finalPrimalError =  1 / (0.0 + instance.total_n) * finalPrimalError
				+ 0.5 * instance.lambda * tmp2 * tmp2;


	}
};






#endif /* SQUAREDHINGELOSS_H_ */
