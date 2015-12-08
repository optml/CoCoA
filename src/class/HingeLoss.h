/*
 * HingeLoss.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef HINGELOSS_H_
#define HINGELOSS_H_

#include "LossFunction.h"

template<typename L, typename D>
class HingeLoss : public LossFunction<L, D> {
public:
	HingeLoss(){

	}
	virtual ~HingeLoss() {}

	virtual void computeObjectiveValue(ProblemData<L, D> & instance,
			mpi::communicator & world, std::vector<D> & w, double &finalDualError,
			double &finalPrimalError){


		D localError = 0;
		for (unsigned int i = 0; i < instance.n; i++) {
			D tmp = -instance.x[i];
			localError += tmp;
		}

		D localHingeLoss = 0;
		for (unsigned int idx = 0; idx < instance.n; idx++) {
			D dotProduct = 0;
			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {
				dotProduct += (w[instance.A_csr_col_idx[i]])
									* instance.A_csr_values[i];
			}
			D tmp = 1 - instance.b[idx] * dotProduct;
			if (tmp > 0) {
				localHingeLoss += tmp;
			}
		}
		finalPrimalError = 0;
		vall_reduce(world, &localHingeLoss, &finalPrimalError, 1);

		finalDualError = 0;
		vall_reduce(world, &localError, &finalDualError, 1);

		D tmp2 = cblas_l2_norm(w.size(), &w[0], 1);
		finalDualError = 1 / (0.0 + instance.total_n) * finalDualError
				+ 0.5 * instance.lambda * tmp2 * tmp2;
		finalPrimalError =  1 / (0.0 + instance.total_n) * finalPrimalError
				+ 0.5 * instance.lambda * tmp2 * tmp2;


	}


	virtual void compute_subproproblem_obj(ProblemData<L, D> &instance,
			std::vector<D> &potent, std::vector<D> &w, D &potent_obj){

		D obj_temp1 = 0.0;
		D obj_temp2 = 0.0;
		D obj_temp3 = 0.0;
		for (unsigned int i = 0; i < instance.n; i++) {
			obj_temp1 += -instance.x[i] - potent[i];
		}

		std::vector<D> Apotent(instance.m);
		cblas_set_to_zero(Apotent);
		vectormatrix_b(potent, instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr,
				instance.b, 1.0, instance.n, Apotent);

		D g_norm = cblas_l2_norm(instance.m, &Apotent[0], 1);
		obj_temp3 = g_norm * g_norm;

		for (L i = 0; i < instance.m; i++)
			obj_temp2 += w[i] * Apotent[i];

		potent_obj = 1.0 / instance.total_n * (obj_temp1 + obj_temp2 + 0.5 * instance.penalty * instance.oneOverLambdaN * obj_temp3);

	}

	virtual void compute_subproproblem_gradient(ProblemData<L, D> &instance,
			std::vector<D> &gradient, std::vector<D> &deltaAlpha, std::vector<D> &w){

		std::vector<D> gradient_temp1(instance.n);
		std::vector<D> gradient_temp2(instance.m);
		std::vector<D> gradient_temp3(instance.n);

		matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr, w, instance.n, gradient_temp1);
		vectormatrix_b(deltaAlpha, instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr,
				instance.b, 1.0, instance.n, gradient_temp2);
		matrixvector(instance.A_csr_values, instance.A_csr_col_idx, instance.A_csr_row_ptr,
				gradient_temp2, instance.n, gradient_temp3);

		for (L i = 0; i < instance.n; i++){
			gradient[i] = 1.0 / instance.total_n * ( gradient_temp1[i] * instance.b[i] - 1.0
					+ 1.0 * instance.penalty * instance.oneOverLambdaN  * gradient_temp3[i] * instance.b[i] );
		}

	}

	virtual void backtrack_linesearch(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &search_direction, std::vector<D> &w, D dualobj,
			D &rho, D &c1ls, D &a, DistributedSettings & distributedSettings){

		int iter = 0;
		std::vector<D> potent(instance.n);

		while (1){

			if (iter > 50){
				a = 0.0;
				break;
			}

			for (L idx = 0; idx < instance.n; idx++){
				potent[idx] = deltaAlpha[idx] - a * search_direction[idx];
				potent[idx] = (potent[idx] > 1 - instance.x[idx]) ? 1 - instance.x[idx]
														 : (potent[idx] < -instance.x[idx] ? -instance.x[idx] : potent[idx]);
			}

			D obj = 0.0;
			this->compute_subproproblem_obj(instance, potent, w, obj);

			D gg;
			gg = cblas_l2_norm(instance.n, &search_direction[0], 1);
			if (obj <= dualobj - c1ls * a * gg * gg){
				for (L idx = 0; idx < instance.n; idx++){
					//deltaAlpha[idx] = (potent[idx] > 1 - instance.x[idx]) ? 1 - instance.x[idx]
					//					 : (potent[idx] < -instance.x[idx] ? -instance.x[idx] : potent[idx]);
					deltaAlpha[idx] = potent[idx];
				}

				dualobj = obj;
				break;
			}
			a = rho * a;
			iter += 1;
		}
	}


	virtual void LBFGS_update(ProblemData<L, D> &instance, std::vector<D> &search_direction, std::vector<D> &old_grad,
			std::vector<D> &sk, std::vector<D> &rk, std::vector<D> &gradient, std::vector<D> &oneoversy,
			L iter_counter, int limit_BFGS, int flag_BFGS) {

		for (L i = 0; i < instance.n; i++){

			if (iter_counter > 0){
				if (flag_BFGS > 0)
					rk[instance.n * (flag_BFGS - 1)  + i] = gradient[i] - old_grad[i];
				else
					rk[instance.n * (limit_BFGS - 1)  + i] = gradient[i] - old_grad[i];
			}

			old_grad[i] = gradient[i];
			search_direction[i] = gradient[i];
		}

		if (iter_counter > 0) {

			int flag_old = flag_BFGS - 1;
			if (flag_BFGS == 0)
				flag_old = limit_BFGS - 1;

			oneoversy[flag_old] = 0;
			for (L idx = 0; idx < instance.n; idx++){
				oneoversy[flag_old] += rk[instance.n * flag_old + idx]* sk[instance.n * flag_old + idx];
			}
			oneoversy[flag_old] = 1.0 / oneoversy[flag_old];

			L kai = flag_BFGS;
			L wan = kai + limit_BFGS - 1;// min(kai + 9, kai + iter_counter - 1);
			std::vector<D> aa(limit_BFGS);
			cblas_set_to_zero(aa);
			D bb = 0.0;
			//cout<<kai<<"    "<<wan<<"    "<<iter_counter<<endl;

			if (iter_counter < limit_BFGS) {
				kai = 0;
				wan = flag_BFGS - 1;
			}

			for (L i = kai; i <= wan; i++){
				L ii = wan - (i-kai);
				if (ii >= limit_BFGS)
					ii = ii - 10;

				for (L j = 0; j < instance.n; j++)
					aa[ii] += sk[instance.n * ii + j] * search_direction[j];
				aa[ii] *= oneoversy[ii];
				for (L idx = 0; idx < instance.n; idx++)
					search_direction[idx] -= aa[ii] * rk[instance.n * ii + idx];
			}
			D Hk_zero = 0.0;
			D t1 = 0.0;
			D t2 = 0.0;
			for (L idx = instance.n * flag_old; idx < instance.n * (flag_old + 1); idx++){
				t1 += sk[idx] * rk[idx];
				t2 += rk[idx] * rk[idx];
				//cout<< t1 << "   "<<t2<<endl;
			}

			Hk_zero = t1 / t2;
			//cout<< Hk_zero << "   "<<t2<<endl;
			for (L idx = 0; idx < instance.n; idx++)
				search_direction[idx] = Hk_zero * search_direction[idx];

			for (L i = kai; i <= wan; i++){
				L ii = i;
				if (i >= limit_BFGS)
					ii = i - limit_BFGS;

				bb = 0.0;
				for (L j = 0; j < instance.n; j++)
					bb += rk[instance.n * ii + j] * search_direction[j];
				bb *= oneoversy[ii];
				for (L j = 0; j < instance.n; j++)
					search_direction[j] += sk[instance.n * ii + j] * (aa[ii] - bb);
			}

		}
	}

};

#endif /* HINGELOSS_H_ */
