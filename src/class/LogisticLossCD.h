/*
 * LogisticLossCD.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef LOGISTICLOSSCD_H_
#define LOGISTICLOSSCD_H_

#include "LogisticLoss.h"
template<typename L, typename D>
class LogisticLossCD :   public LogisticLoss<L, D>{
public:
	LogisticLossCD(){

	}
	virtual ~LogisticLossCD() {}

	virtual void subproblem_solver_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {

			start = gettime_();

			for (int jj = 0; jj < distributedSettings.iters_bulkIterations_count; jj++) {
				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
						it++) {

					L idx = rand() / (0.0 + RAND_MAX) * instance.n;

					D dotProduct = 0;
					for (L i = instance.A_csr_row_ptr[idx];
							i < instance.A_csr_row_ptr[idx + 1]; i++) {

						dotProduct += (w[instance.A_csr_col_idx[i]]
										 + 1.0 * instance.penalty * deltaW[instance.A_csr_col_idx[i]])
																	* instance.A_csr_values[i];
					}

					D alphaI = instance.x[idx] + deltaAlpha[idx];

					D norm = cblas_l2_norm(
							instance.A_csr_row_ptr[idx + 1] - instance.A_csr_row_ptr[idx],
							&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);

					D deltaAl = 0.0;
					D epsilon = 1e-5;

					if (alphaI == 0) {deltaAl = 0.1 * instance.b[idx];}
					D FirstDerivative = 1.0 * instance.penalty * deltaAl * instance.oneOverLambdaN * norm * norm
							+ dotProduct * instance.b[idx] - log(1.0 - (alphaI + deltaAl) / instance.b[idx]) / instance.b[idx]
                            + log((alphaI + deltaAl) / instance.b[idx]) / instance.b[idx];

					while (FirstDerivative > epsilon || FirstDerivative < -1.0 * epsilon)
					{
						D SecondDerivative = 1.0 * instance.penalty * norm * norm * instance.oneOverLambdaN
								+ 1.0 / (1.0 - (alphaI + deltaAl) / instance.b[idx]) + 1.0 / (alphaI + deltaAl) / instance.b[idx];
						deltaAl = 1.0 * deltaAl - FirstDerivative / SecondDerivative;

						if (instance.b[idx] == 1.0)
							deltaAl = (deltaAl > 1 - alphaI) ? 1 - alphaI - 1e-15 : (deltaAl < -alphaI ? -alphaI + 1e-15 : deltaAl);
						else if (instance.b[idx] == -1.0)
							deltaAl = (deltaAl > -alphaI) ? -alphaI - 1e-15 : (deltaAl < -1.0 - alphaI ? -1.0 - alphaI + 1e-15: deltaAl);
						//if ((alphaI+ deltaAl)/instance.b[idx] == -1) cout<<idx<<endl;
						FirstDerivative = 1.0 * instance.penalty * deltaAl * instance.oneOverLambdaN * norm * norm
								+ dotProduct * instance.b[idx] - log(1.0 - (alphaI + deltaAl) / instance.b[idx]) / instance.b[idx]
	                            + log((alphaI + deltaAl) / instance.b[idx]) / instance.b[idx];
					}

//					if (instance.b[idx] == 1.0)
//					{
//						if (alphaI == 0) {deltaAl = 0.5;}
//						D FirstDerivative = 1.0 * instance.penalty * deltaAl * instance.oneOverLambdaN * norm * norm
//								+ dotProduct * instance.b[idx] - log(1.0 - alphaI - deltaAl) + log(alphaI + deltaAl);
//
//						while (FirstDerivative > epsilon || FirstDerivative < -epsilon)
//						{
//							D SecondDerivative = 1.0 * instance.penalty * norm * norm * instance.oneOverLambdaN
//									+ 1.0 / (1.0 - alphaI - deltaAl) + 1.0 / (alphaI + deltaAl);
//							deltaAl = 1.0 * deltaAl - FirstDerivative / SecondDerivative;
//							deltaAl = (deltaAl > 1 - alphaI) ? 1 - alphaI - 1e-15 : (deltaAl < -alphaI ? -alphaI + 1e-15 : deltaAl);
//							FirstDerivative = 1.0 * instance.penalty * deltaAl * instance.oneOverLambdaN * norm * norm
//									+ dotProduct * instance.b[idx] - log(1.0 - alphaI - deltaAl) + log(alphaI + deltaAl);
//						}
//						//cout<<deltaAl+alphaI<<"  ";
//						//cout<<FirstDerivative<<"  ";
//					}
//
//					else if (instance.b[idx] == -1.0)
//					{
//						if(alphaI == 0) {deltaAl = -0.5;}
//						D FirstDerivative = 1.0 * instance.penalty * deltaAl * instance.oneOverLambdaN * norm * norm
//								+ dotProduct * instance.b[idx] + log(1.0 + alphaI + deltaAl) - log(-1.0 * alphaI - deltaAl);
//
//						while (FirstDerivative > epsilon || FirstDerivative < -epsilon)
//						{
//							D SecondDerivative = 1.0 * instance.penalty * norm * norm * instance.oneOverLambdaN
//									+ 1.0 / (1.0 + alphaI + deltaAl) - 1.0 / (alphaI + deltaAl);
//							deltaAl = 1.0 * deltaAl - FirstDerivative / SecondDerivative;
//							deltaAl = (deltaAl > -alphaI) ? -alphaI - 1e-15 : (deltaAl < -1.0 - alphaI ? -1.0 - alphaI + 1e-15: deltaAl);
//							FirstDerivative = 1.0 * instance.penalty * deltaAl * instance.oneOverLambdaN * norm * norm
//									+ dotProduct * instance.b[idx] + log(1.0 + alphaI + deltaAl) - log(-1.0 * alphaI - deltaAl);
//							//if(idx==52) cout<<idx<<"  1  "<<deltaAl<<"  2  "<<FirstDerivative<<"  3  "<<SecondDerivative<<"  5  "<<alphaI<<"  6  "<<log(1.0+alphaI+deltaAl)<<endl;
//						}
//						//cout<<deltaAl+alphaI<<"  ";
//						//cout<<FirstDerivative<<"  ";
//					}
					deltaAlpha[idx] += deltaAl;
					for (L i = instance.A_csr_row_ptr[idx];
							i < instance.A_csr_row_ptr[idx + 1]; i++) {

						D tmd =  instance.oneOverLambdaN * instance.A_csr_values[i] * deltaAl * instance.b[idx];
						deltaW[instance.A_csr_col_idx[i]] += tmd;
					}
				}
				//cout<<deltaAlpha[148]<<endl;
				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime
						<< "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << ","
						<< dualError << "," << primalError + dualError << endl;

			}
		}
	}




	virtual void accelerated_SDCA_oneIteration(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w, std::vector<D> &deltaW,
			std::vector<D> &zk, std::vector<D> &uk,
			std::vector<D> &yk, std::vector<D> &deltayk, std::vector<D> &Ayk, std::vector<D> &deltaAyk,
			D &theta, DistributedSettings & distributedSettings){

		D thetasquare = theta * theta;
		L idx = rand() / (0.0 + RAND_MAX) * instance.n;

		D dotProduct = 0;
		for (L i = instance.A_csr_row_ptr[idx]; i < instance.A_csr_row_ptr[idx + 1]; i++) {
			dotProduct += (1.0 * Ayk[instance.A_csr_col_idx[i]] + instance.penalty * deltaAyk[instance.A_csr_col_idx[i]])
																										* instance.A_csr_values[i];
		}

		D tk = ( 1.0 * instance.b[idx] - zk[idx] - dotProduct * instance.b[idx] ) //- thetasquare * uk[idx])
													/ (instance.vi[idx] * instance.n /distributedSettings.iterationsPerThread
															* theta * instance.penalty * instance.oneOverLambdaN + 1.);
		zk[idx] += tk;
		uk[idx] -= (1.0 - theta * instance.n /distributedSettings.iterationsPerThread) / (thetasquare) * tk;

		D deltaAl = thetasquare * uk[idx] + zk[idx] - instance.x[idx] - deltaAlpha[idx];
		deltaAlpha[idx] += deltaAl;

		D thetanext = theta;
		thetanext = 0.5 * sqrt(thetasquare * thetasquare + 4 * thetasquare) - 0.5 * thetasquare;
		D dyk = thetanext * thetanext * uk[idx] + zk[idx] - yk[idx] - deltayk[idx];
		deltayk[idx] += dyk;

	}

	virtual void subproblem_solver_accelerated_SDCA(ProblemData<L, D> &instance, std::vector<D> &deltaAlpha,
			std::vector<D> &w, std::vector<D> &wBuffer, std::vector<D> &deltaW, DistributedSettings & distributedSettings,
			mpi::communicator &world, D gamma, Context &ctx, std::ofstream &logFile) {

		double start = 0;
		double finish = 0;
		double elapsedTime = 0;

		double theta = 1.0 * distributedSettings.iterationsPerThread / instance.n;
		std::vector<double> zk(instance.n);
		std::vector<double> uk(instance.n);
		std::vector<double> Ayk(instance.m);
		std::vector<double> yk(instance.n);
		std::vector<double> deltayk(instance.n);
		std::vector<double> deltaAyk(instance.m);
		cblas_set_to_zero(uk);
		cblas_set_to_zero(yk);
		cblas_set_to_zero(Ayk);
		std::vector<double> AykBuffer(instance.m);

		for (unsigned int i = 0; i < instance.n; i++)
			zk[i] = instance.x[i];

		for (unsigned int t = 0; t < distributedSettings.iters_communicate_count; t++) {
			start = gettime_();
			for (int jj = 0; jj < 10*distributedSettings.iters_bulkIterations_count; jj++) {

				cblas_set_to_zero(deltaW);
				cblas_set_to_zero(deltaAlpha);
				cblas_set_to_zero(deltayk);
				cblas_set_to_zero(deltaAyk);

				for (unsigned int it = 0; it < distributedSettings.iterationsPerThread; it++)
					this->accelerated_SDCA_oneIteration(instance, deltaAlpha, w, deltaW,
							zk, uk, yk, deltayk, Ayk, deltaAyk, theta, distributedSettings);

				double thetasq = theta * theta;
				theta = 0.5 * sqrt(thetasq * thetasq + 4 * thetasq) - 0.5 * thetasq;
				cblas_sum_of_vectors(instance.x, deltaAlpha, gamma);
				cblas_sum_of_vectors(yk, deltayk, gamma);

				vectormatrix_b(instance.x, instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr,
						instance.b, instance.oneOverLambdaN, instance.n, deltaW);

				vectormatrix_b(yk, instance.A_csr_values, instance.A_csr_col_idx,instance.A_csr_row_ptr,
						instance.b, instance.oneOverLambdaN, instance.n, deltaAyk);

				cblas_set_to_zero(w);
				cblas_set_to_zero(Ayk);

				vall_reduce(world, deltaW, wBuffer);
				cblas_sum_of_vectors(w, wBuffer, gamma);
				vall_reduce(world, deltaAyk, AykBuffer);
				cblas_sum_of_vectors(Ayk, AykBuffer, gamma);

			}
			double primalError;
			double dualError;

			finish = gettime_();
			elapsedTime += finish - start;

			this->computeObjectiveValue(instance, world, w, dualError, primalError);

			if (ctx.settings.verbose) {
				cout << "Iteration " << t << " elapsed time " << elapsedTime << "  error " << primalError << "    " << dualError
						<< "    " << primalError + dualError << endl;

				logFile << t << "," << elapsedTime << "," << primalError << "," << dualError << "," << primalError + dualError << endl;
			}


		}

	}
};

#endif /* LOGISTICLOSSCD_H_ */
