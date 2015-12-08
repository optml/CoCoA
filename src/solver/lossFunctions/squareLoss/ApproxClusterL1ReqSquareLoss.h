/*
 * ClusterL1ReqSquadeLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef APPROXCLUSTERL1REQSQUADELOSS_H_
#define APPROXCLUSTERL1REQSQUADELOSS_H_

#include "ClusterL1ReqSquareLoss.h"
namespace Loss {
template<typename L, typename D>
class ApproxClusterL1ReqSquareLoss: public Loss::ClusterL1ReqSquareLoss<L, D> {
public:

	ApproxClusterL1ReqSquareLoss(mpi::communicator *_world,
			ProblemData<L, D>* _localInstance, std::vector<D>* _localResiduals,
			ProblemData<L, D>* _globalInstance,
			std::vector<D>* _globalResiduals, std::vector<D>* _residualsBuffer,
			D* _residual_updatesPtr) :
			ClusterL1ReqSquareLoss<L, D>(_world, _localInstance,
					_localResiduals, _globalInstance, _globalResiduals,
					_residualsBuffer, _residual_updatesPtr) {
	}

	void recomputeResiduals() {
		ProblemData<L, D>& instLocal = *this->localInstance;
		ProblemData<L, D>& instGlob = *this->globalInstance;

		std::vector < D > bck(instLocal.m * 2);
		for (L i = 0; i < instLocal.m * 2; i++) {
			bck[i] = (*this->localResiduals)[i];
		}

		// set r=-b+A*x on local data
		for (L i = 0; i < instLocal.m; i++) {
			(*this->localResiduals)[i] = -instLocal.b[i];
			(*this->localResiduals)[i + instLocal.m] = 0;
		}
		for (L i = 0; i < instLocal.n; i++) {
			for (L j = instLocal.A_csc_col_ptr[i];
					j < instLocal.A_csc_col_ptr[i + 1]; j++) {
				(*this->localResiduals)[instLocal.A_csc_row_idx[j]] +=
						instLocal.A_csc_values[j] * instLocal.z[i];
				(*this->localResiduals)[instLocal.A_csc_row_idx[j] + instLocal.m] +=
						instLocal.A_csc_values[j] * instLocal.u[i];
			}
		}

// set r=A*x on global data
		for (L i = 0; i < 2 * instGlob.m; i++) {
			(*this->residualsBuffer)[i] = 0;
		}
		for (L i = 0; i < instGlob.n; i++) {
			for (L j = instGlob.A_csc_col_ptr[i];
					j < instGlob.A_csc_col_ptr[i + 1]; j++) {
				(*this->residualsBuffer)[instGlob.A_csc_row_idx[j]] +=
						instGlob.A_csc_values[j] * instLocal.z[i];
				(*this->residualsBuffer)[instGlob.A_csc_row_idx[j] + instGlob.m] +=
						instGlob.A_csc_values[j] * instLocal.u[i];
			}
		}
		// reduce all
		vall_reduce(*this->world, *this->residualsBuffer,
				*this->globalResiduals);
		for (L i = 0; i < instGlob.m; i++) {
			(*this->globalResiduals)[i] -= instGlob.b[i];
		}

	}

	virtual ~ApproxClusterL1ReqSquareLoss() {

	}

	virtual inline void performOneCoordinateUpdate(L idx) {
		ProblemData<L, D>& instLocal = *this->localInstance;
		ProblemData<L, D>& instGlob = *this->globalInstance;

		D gamma = instLocal.total_n * instLocal.theta / instLocal.Li[idx]
				/ (0.0 + instLocal.total_tau);
		D thetaSq = instLocal.theta * instLocal.theta;
		D gradient = 0;
		for (L j = instLocal.A_csc_col_ptr[idx];
				j < instLocal.A_csc_col_ptr[idx + 1]; j++) {
			gradient +=
					instLocal.A_csc_values[j]
							* ((*this->localResiduals)[instLocal.A_csc_row_idx[j]]
									+ thetaSq
											* (*this->localResiduals)[instLocal.A_csc_row_idx[j]
													+ instLocal.m]);
		}
		for (L j = instGlob.A_csc_col_ptr[idx];
				j < instGlob.A_csc_col_ptr[idx + 1]; j++) {
			gradient +=
					instGlob.A_csc_values[j]
							* ((*this->globalResiduals)[instGlob.A_csc_row_idx[j]]
									+ thetaSq
											* (*this->globalResiduals)[instGlob.A_csc_row_idx[j]
													+ instGlob.m]);
		}

		D tmp = compute_soft_treshold(instLocal.lambda / gamma,
				instLocal.z[idx] - gradient / gamma) - instLocal.z[idx];

		D du = -(1
				- instLocal.total_n / (0.0 + instLocal.total_tau)
						* instLocal.theta) / thetaSq;

//		cout << tmp << " "<< du << " " << thetaSq << " " << (1 - instLocal.n / (0.0 + instLocal.tau)) << endl;

		parallel::atomic_add(instLocal.z[idx], tmp);
		parallel::atomic_add(instLocal.u[idx], du * tmp);

		for (L j = instLocal.A_csc_col_ptr[idx];
				j < instLocal.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(
					(*this->localResiduals)[instLocal.A_csc_row_idx[j]],
					tmp * instLocal.A_csc_values[j]);

			parallel::atomic_add(
					(*this->localResiduals)[instLocal.A_csc_row_idx[j]
							+ instLocal.m],
					du * tmp * instLocal.A_csc_values[j]);

		}
		for (L j = instGlob.A_csc_col_ptr[idx];
				j < instGlob.A_csc_col_ptr[idx + 1]; j++) {
			parallel::atomic_add(
					this->residual_updatesPtr[instGlob.A_csc_row_idx[j]],
					tmp * instGlob.A_csc_values[j]);

			parallel::atomic_add(
					this->residual_updatesPtr[instGlob.A_csc_row_idx[j]
							+ instGlob.m], du * tmp * instGlob.A_csc_values[j]);

		}
	}

	D computeObjectiveValue() {
		ProblemData<L, D>& instLocal = *this->localInstance;
		ProblemData<L, D>& instGlob = *this->globalInstance;

		D localRedidsSq = 0;
		D globalRedidsSq = 0;
		D localRedidsSqOut = 0;

		D localL1Norm = 0;
		D thetaSq = instLocal.prevTheta * instLocal.prevTheta;

		for (L i = 0; i < instLocal.m; i++) {
			D tmp = (*this->localResiduals)[i]
					+ (*this->localResiduals)[i + instLocal.m] * thetaSq;
			localRedidsSq += tmp * tmp;
		}
		for (L i = 0; i < instGlob.m; i++) {
			D tmp = (*this->globalResiduals)[i]
					+ (*this->globalResiduals)[i + instGlob.m] * thetaSq;
			globalRedidsSq += tmp * tmp;
		}

		for (L i = 0; i < instLocal.n; i++) {
			D tmp = instLocal.z[i] + thetaSq * instLocal.u[i];
			localL1Norm += abs(tmp);
		}

		D l1NormOut = 0;
		reduce(*this->world, localRedidsSq, localRedidsSqOut, std::plus<D>(),
				0);
		reduce(*this->world, localL1Norm, l1NormOut, std::plus<D>(), 0);
		this->lastComputedObjectiveValue = 0.5
				* (globalRedidsSq + localRedidsSqOut)
				+ instLocal.lambda * l1NormOut;
		return this->lastComputedObjectiveValue;
	}

};
}

#endif /* CLUSTERL1REQSQUADELOSS_H_ */
