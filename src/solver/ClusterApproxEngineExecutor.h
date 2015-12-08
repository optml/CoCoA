/*
 * MulticoreEngineExecutor.h
 *
 *  Created on: Sep 10, 2013
 *      Author: taki
 */

#ifndef CLUSTERAPPROXENGINEEXECUTOR_H_
#define CLUSTERAPPROXENGINEEXECUTOR_H_
#include "ClusterEngineExecutor.h"
#include "./structures.h"
#include "settingsAndStatistics.h"

#include "lossFunctions/approxClusterLossFunctions.h"
#include "../utils/randomNumbersUtil.h"
#include "distributed/distributed_structures.h"
#include "distributed/distributed_synchronous.h"
#include "distributed/distributed_asynchronous.h"
template<typename L, typename D>
class ClusterApproxEngineExecutor: public ClusterEngineExecutor<L, D> {

public:

	ClusterApproxEngineExecutor(mpi::communicator &_world,
			ProblemData<L, D> & _localInstance,
			ProblemData<L, D> & _globalInstance,
			DistributedSettings* _distributedSettings) :
			ClusterEngineExecutor<L, D>(_world, _localInstance, _globalInstance,
					_distributedSettings) {
		this->localInstance.theta = _localInstance.total_tau
				/ (0.0 + _localInstance.total_n);

		this->localInstance.prevTheta = this->localInstance.theta;

		this->localInstance.z.resize(this->localInstance.n);
		this->localInstance.u.resize(this->localInstance.n);
		cblas_vector_scale(this->localInstance.n, &this->localInstance.z[0],
				(D) 0);
		cblas_vector_scale(this->localInstance.n, &this->localInstance.u[0],
				(D) 0);

		// first part of residuals correcponds to vector "z" and the rest to vector "u"

		this->localResiduals.resize(2 * this->localInstance.m);
		this->globalResiduals.resize(2 * this->globalInstance.m);
		this->residualsBuffer.resize(2 * this->globalInstance.m);
		this->residual_updates_to_exchange.resize(2 * this->globalInstance.m);

	}

	virtual void initializeBuffers() {

		this->residual_updates.resize(2 * this->globalInstance.m, 0);
		this->residual_updatesPtr = &this->residual_updates[0];

		if (this->distributedSettings->distributed == SynchronousReduce) {
			this->exchanged.resize(this->globalInstance.m * 2);
		}

//		if (distributedSettings->distributed
//				== AsynchronousStreamlinedOptimized) {
//			past_update.resize(globalInstance.m * (world.size()), 0);
//			exchanged.resize(globalInstance.m * 2, 0);
//			exchange_data = 0;
//
//		}
	}

	virtual void initializeLossFunction() {

		switch (this->distributedSettings->lossFunction) {
		case 0:
			this->loss = new Loss::ApproxClusterL1ReqSquareLoss<L, D>(
					&this->world, &this->localInstance, &this->localResiduals,
					&this->globalInstance, &this->globalResiduals,
					&this->residualsBuffer, this->residual_updatesPtr);
			break;
		case 1:
//			this->loss = new Loss::ClusterL2SquareReqSquareLoss<L, D>(&this->world,
//					&this->localInstance, &this->localResiduals, &this->globalInstance,
//					&this->globalResiduals, &this->residualsBuffer, this->residual_updatesPtr);
//			break;

		case 2:
			this->loss = new Loss::ApproxClusterSVMDualLoss<L, D>(this->world,
					this->localInstance, this->localResiduals,
					this->globalInstance, this->globalResiduals,
					this->residualsBuffer, this->residual_updatesPtr);
			break;

		default:
			break;
		}
	}
	void getNewTheta() {
		D tmp = this->localInstance.theta * this->localInstance.theta;
		this->localInstance.prevTheta = this->localInstance.theta;
		this->localInstance.theta = (sqrt(tmp * tmp + 4 * tmp) - tmp) / 2;

	}

	virtual void executeBulkOfIterations() {
		L innerIterations = this->distributedSettings->iterationsPerThread
				* TOTAL_THREADS;
		for (unsigned int i = 0;
				i < this->distributedSettings->iters_communicate_count; i++) {
			cblas_set_to_zero(2 * this->globalInstance.m,
					this->residual_updatesPtr);
			{
#pragma omp parallel for
//				for (int i = 0; i < innerIterations; i++) {
				for (L i = 0; i < this->localInstance.n; i++) {
//					L idx = gsl_rng_uniform_int(gsl_rng_r,
//							this->localInstance.n);
					L idx=i;
					this->loss->performOneCoordinateUpdate(idx);
				}
			}

			this->exchange_residuals(this->residual_updates);

			this->getNewTheta();
		}
	}

	virtual ~ClusterApproxEngineExecutor() {
	}

};

#endif /* CLUSTERENGINEEXECUTOR_H_ */
