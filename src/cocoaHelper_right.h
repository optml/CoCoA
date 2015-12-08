/*~OA
 * cocoaHelper.h
 *
 *  Created on: Nov 17, 2014
 *      Author: taki
 */
#ifndef COCOAHELPER_H_
#define COCOAHELPER_H_

#include <iomanip>
using namespace std;

template<typename L, typename D>
void localCDMethodHingeLoss(ProblemData<L, D> &instance,
		std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
		DistributedSettings & distributedSettings) {

	for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
			it++) {

		L idx = rand() / (0.0 + RAND_MAX) * instance.n;

// compute "delta alpha" = argmin

		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {

			dotProduct += (w[instance.A_csr_col_idx[i]]
					+ deltaW[instance.A_csr_col_idx[i]])
					* instance.A_csr_values[i];

		}

		D alphaI = instance.x[idx] + deltaAlpha[idx];

		D deltaAl = 0; // FINISH

		D norm = cblas_l2_norm(
				instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
				&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);

		D part = (instance.lambda * instance.total_n)
				* (1.0 - instance.b[idx] * dotProduct) / (norm * norm);

		deltaAl =
				(part > 1 - alphaI) ?
						1 - alphaI : (part < -alphaI ? -alphaI : part);

		deltaAlpha[idx] += deltaAl;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {

			deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
					* instance.A_csr_values[i] * deltaAl * instance.b[idx];

		}

	}

}

template<typename L, typename D>
void localCDMethodQuadLoss(ProblemData<L, D> &instance,
		std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
		DistributedSettings & distributedSettings) {

	for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
			it++) {

		L idx = rand() / (0.0 + RAND_MAX) * instance.n;

// compute "delta alpha" = argmin

		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {

			dotProduct += (w[instance.A_csr_col_idx[i]]
					+ deltaW[instance.A_csr_col_idx[i]])
					* instance.A_csr_values[i];

		}

		D alphaI = instance.x[idx] + deltaAlpha[idx];

		D deltaAl = 0; // FINISH

		D norm = cblas_l2_norm(
				instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
				&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);

		D denom = norm * norm / instance.lambda / instance.n + 0.5;

		deltaAl = (1.0 - 0.5 * alphaI - instance.b[idx] * dotProduct) / denom;

		deltaAlpha[idx] += deltaAl;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {

			deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
					* instance.A_csr_values[i] * deltaAl * instance.b[idx];

		}

	}

}

template<typename L, typename D>
void localCDMethodSquaredHingeLoss(ProblemData<L, D> &instance,
		std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
		DistributedSettings & distributedSettings) {

	for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
			it++) {

		L idx = rand() / (0.0 + RAND_MAX) * instance.n;

// compute "delta alpha" = argmin

		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {

			dotProduct += (w[instance.A_csr_col_idx[i]]
					+ deltaW[instance.A_csr_col_idx[i]])
					* instance.A_csr_values[i];

		}

		D alphaI = instance.x[idx] + deltaAlpha[idx];

		D deltaAl = 0; // FINISH

		D norm = cblas_l2_norm(
				instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
				&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);

		D denom = norm * norm / instance.lambda / instance.n + 0.5;

		deltaAl = (1.0 - 0.5 * alphaI - instance.b[idx] * dotProduct) / denom;

		deltaAl = (deltaAl < -alphaI) ? -alphaI : deltaAl;

		deltaAlpha[idx] += deltaAl;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {

			deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
					* instance.A_csr_values[i] * deltaAl * instance.b[idx];

		}

	}

}

template<typename L, typename D>
void localCDMethodLogisticLoss(ProblemData<L, D> &instance,
		std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
		DistributedSettings & distributedSettings) {

	for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
			it++) {

		L idx = rand() / (0.0 + RAND_MAX) * instance.n;
		
// compute "delta alpha" = argmin

		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {

			dotProduct += (w[instance.A_csr_col_idx[i]]
					+ deltaW[instance.A_csr_col_idx[i]])
					* instance.A_csr_values[i];

		}
		//cout<<deltaAlpha[idx];

		D alphaI = instance.x[idx] + deltaAlpha[idx];

		D norm = cblas_l2_norm(
				instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
				&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);

		dotProduct = instance.b[idx] * dotProduct;

		D deltaAl = 0.0;
		D epsilon = 1e-12;
		if (instance.b[idx] == 1.0)
		{
		        if (alphaI == 0) {deltaAl = 0.5;}
		        D FirstDerivative = 1.0 * deltaAl * instance.oneOverLambdaN * norm * norm
				  + dotProduct - log(1.0 - alphaI - deltaAl) + log(alphaI + deltaAl);

			while (FirstDerivative > epsilon || FirstDerivative < -epsilon)
			{
				D SecondDerivative = 1.0 * norm * norm * instance.oneOverLambdaN
						+ 1.0 / (1.0 - alphaI - deltaAl) + 1.0 / (alphaI + deltaAl);
				deltaAl = 1.0 * deltaAl - FirstDerivative / SecondDerivative;
				deltaAl = (deltaAl > 1 - alphaI) ? 1 - alphaI - 1e-15 : (deltaAl < -alphaI ? -alphaI + 1e-15 : deltaAl);
				FirstDerivative = 1.0 * deltaAl * instance.oneOverLambdaN * norm * norm
					  + dotProduct - log(1.0 - alphaI - deltaAl) + log(alphaI + deltaAl);
			}
		        //cout<<deltaAl+alphaI<<"  ";
			//cout<<FirstDerivative<<"  ";
		}

		else if (instance.b[idx] == -1.0)
		    {
   		        if(alphaI == 0) {deltaAl = -0.5;}
		        D FirstDerivative = 1.0 * deltaAl * instance.oneOverLambdaN * norm * norm
					+ dotProduct + log(1.0 + alphaI + deltaAl) - log(-1.0 * alphaI - deltaAl);

			while (FirstDerivative > epsilon || FirstDerivative < -epsilon)
			{
				D SecondDerivative = norm * norm * instance.oneOverLambdaN
						+ 1.0 / (1.0 + alphaI + deltaAl) - 1.0 / (alphaI + deltaAl);
				deltaAl = 1.0 * deltaAl - FirstDerivative / SecondDerivative;
				deltaAl = (deltaAl > -alphaI) ? -alphaI - 1e-15 : (deltaAl < -1.0 - alphaI ? -1.0 - alphaI + 1e-15: deltaAl);
				FirstDerivative = 1.0* deltaAl * instance.oneOverLambdaN * norm * norm
						+ dotProduct + log(1.0 + alphaI + deltaAl) - log(-1.0 * alphaI - deltaAl);
				//if(idx==52) cout<<idx<<"  1  "<<deltaAl<<"  2  "<<FirstDerivative<<"  3  "<<SecondDerivative<<"  5  "<<alphaI<<"  6  "<<log(1.0+alphaI+deltaAl)<<endl;
			}
			//cout<<deltaAl+alphaI<<"  ";
			//cout<<FirstDerivative<<"  ";
		}
		//if (isnan(deltaAl)) {cout<<deltaAl<<" 1 "<<alphaI<<"  2  "<<instance.b[idx]<<"  3  "<<idx<<endl;}
		//cout<<deltaAl<<"  ";
		//cout<<idx<<"  ";
		deltaAlpha[idx] += deltaAl;
		//L mm=1;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {
		        //cout<<deltaW[instance.A_csr_col_idx[i]]<<"  ";

		        D tmd =  instance.oneOverLambdaN * instance.A_csr_values[i] * deltaAl * instance.b[idx];
			//if(fabs(tmd) <eps) tmd = 0;
			//D ctmd = deltaW[instance.A_csr_col_idx[i]];
			deltaW[instance.A_csr_col_idx[i]] += tmd;
     			/*if (isnan(tmd) && mm==1) 
			  {
			    mm=0;
			    cout<<tmd<<"  "<<deltaAl <<"      "<<idx <<"           "<<instance.A_csr_col_idx[i]<<endl;
			    }*/
			//cout <<instance.A_csr_col_idx[i]<<"  "<<idx<<endl;}
			//if (instance.A_csr_col_idx[i] == 5) cout<< tmd<<"  "<<deltaW[5]<<endl;
			//cout<<deltaW[instance.A_csr_col_idx[i]]<<"      ";			
			//cout<<setprecision (16) <<tmd<<"  ";
		}		      		
	}

}


template<typename L, typename D>
void localLBFGSMethod(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
		std::vector<D> &w, std::vector<D> &deltaW,
		DistributedSettings & distributedSettings) {

}

template<typename L, typename D>
void computeObjectiveValueHingeLoss(ProblemData<L, D> & instance,
		mpi::communicator & world, std::vector<D> & w, double &finalDualError,
		double &finalPrimalError) {

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

template<typename L, typename D>
void computeObjectiveValueQuadLoss(ProblemData<L, D> & instance,
		mpi::communicator & world, std::vector<D> & w, double &finalDualError,
		double &finalPrimalError) {

	D localError = 0;
	for (unsigned int i = 0; i < instance.n; i++) {
		D tmp = instance.x[i] * instance.x[i] / 4 - instance.x[i];
		localError += tmp;
	}

	D localQuadLoss = 0;
	for (unsigned int idx = 0; idx < instance.n; idx++) {
		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct += (w[instance.A_csr_col_idx[i]])
					* instance.A_csr_values[i];
		}
		D tmp = (1 - instance.b[idx] * dotProduct) * (1 - instance.b[idx] * dotProduct);

		localQuadLoss += tmp;

	}
	finalPrimalError = 0;
	vall_reduce(world, &localQuadLoss, &finalPrimalError, 1);

	finalDualError = 0;
	vall_reduce(world, &localError, &finalDualError, 1);

	D tmp2 = cblas_l2_norm(w.size(), &w[0], 1);
	finalDualError = 1 / (0.0 + instance.total_n) * finalDualError
			+ 0.5 * instance.lambda * tmp2 * tmp2;
finalPrimalError =  1 / (0.0 + instance.total_n) * finalPrimalError
		+ 0.5 * instance.lambda * tmp2 * tmp2;

}

template<typename L, typename D>
void computeObjectiveValueSquaredHingeLoss(ProblemData<L, D> & instance,
		mpi::communicator & world, std::vector<D> & w, double &finalDualError,
		double &finalPrimalError) {

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

template<typename L, typename D>
void computeObjectiveValueLogisticLoss(ProblemData<L, D> & instance,
		mpi::communicator & world, std::vector<D> & w, double &finalDualError,
		double &finalPrimalError) {

	D localError = 0;
	for (unsigned int i = 0; i < instance.n; i++) {
		D tmp = 0;
		//if(instance.b[i]*instance.x[i]<=0)cout<<instance.b[i]<<"  "<<instance.x[i]<<endl;
		if (instance.b[i] == -1.0){
		  //cout<<instance.x[i]<<endl;
			if (instance.x[i] < 0){
				tmp += -instance.x[i] * log(-instance.x[i]) ;
			}
			if (instance.x[i] > -1){
				tmp += (1 + instance.x[i]) * log(1 + instance.x[i]);
			}

		}
		if (instance.b[i] == 1.0){
		  //cout<<instance.x[i]<<"   ";
		        if (instance.x[i] > 0){
				tmp += instance.x[i] * log(instance.x[i]) ;
			}
			if (instance.x[i] < 1){
				tmp += (1 - instance.x[i]) * log(1 - instance.x[i]);
			}
		}
		//if (tmp>0) cout<<tmp<<endl;
		localError += tmp;
	}
	//cout<<localError<<" ";

	D localLogisticLoss = 0;
	for (unsigned int idx = 0; idx < instance.n; idx++) {
		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx];
				i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct += (w[instance.A_csr_col_idx[i]])
					* instance.A_csr_values[i];
		}
		
		D tmp = -1.0 * dotProduct;		
		localLogisticLoss += log(1 + exp(tmp));
	        
	}
	//cout<<localLogisticLoss<<"  ";
	finalPrimalError = 0;
	vall_reduce(world, &localLogisticLoss, &finalPrimalError, 1);

	finalDualError = 0;
	vall_reduce(world, &localError, &finalDualError, 1);

	D tmp2 = cblas_l2_norm(w.size(), &w[0], 1);
	finalDualError = 1.0 / instance.total_n * finalDualError
			+ 0.5 * instance.lambda * tmp2 * tmp2;
	finalPrimalError =  1.0 / instance.total_n * finalPrimalError
		+ 0.5 * instance.lambda * tmp2 * tmp2;
	//for(unsigned int i=0; i<w.size();i++){cout<<w[i]<<"  ";}

}


#endif /* COCOAHELPER_H_ */

