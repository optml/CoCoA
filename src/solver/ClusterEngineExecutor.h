/*
 * MulticoreEngineExecutor.h
 *
 *  Created on: Sep 10, 2013
 *      Author: taki
 */

#ifndef CLUSTERENGINEEXECUTOR_H_
#define CLUSTERENGINEEXECUTOR_H_
#include "AbstractEngineExecutor.h"
#include "./structures.h"
#include "settingsAndStatistics.h"

#include "lossFunctions/clusterLossFunctions.h"
#include "../utils/randomNumbersUtil.h"
#include "distributed/distributed_structures.h"
#include "distributed/distributed_synchronous.h"
#include "distributed/distributed_asynchronous.h"
template<typename L, typename D>
class ClusterEngineExecutor: public AbstractEngineExecutor<L, D> {

public:
	ProblemData<L, D> & localInstance;
	ProblemData<L, D> & globalInstance;
	Loss::AbstractLoss<L, D>* loss;
	std::vector<D> localResiduals;
	std::vector<D> globalResiduals;
	std::vector<D> residualsBuffer;
	std::vector<D> residual_updates;
	std::vector<D> residual_updates_to_exchange;
	D* residual_updatesPtr;
	int AsynchronousStreamlinedOptimizedUpdate;
	std::vector<gsl_rng *> rs;
	mpi::communicator &world;
	DistributedSettings* distributedSettings;

	virtual void initializeLossFunction() {
		switch (distributedSettings->lossFunction) {
		case 0:
			this->loss = new Loss::ClusterL1ReqSquareLoss<L, D>(&world,
					&localInstance, &localResiduals, &globalInstance,
					&globalResiduals, &residualsBuffer, residual_updatesPtr);
			break;
		case 1:
			this->loss = new Loss::ClusterL2SquareReqSquareLoss<L, D>(&world,
					&localInstance, &localResiduals, &globalInstance,
					&globalResiduals, &residualsBuffer, residual_updatesPtr);
			break;

		case 2:
			this->loss = new Loss::ClusterSVMDualLoss<L, D>(world,
					localInstance, localResiduals, globalInstance,
					globalResiduals, residualsBuffer, residual_updatesPtr);
			break;

		default:
			break;
		}
	}

	ClusterEngineExecutor(mpi::communicator &_world,
			ProblemData<L, D> & _localInstance,
			ProblemData<L, D> & _globalInstance,
			DistributedSettings* _distributedSettings) :
			world(_world), localInstance(_localInstance), globalInstance(
					_globalInstance) {
		this->AsynchronousStreamlinedOptimizedUpdate = 0;
		distributedSettings = _distributedSettings;
		this->settings = distributedSettings;
		localInstance.x.resize(localInstance.n);
		cblas_vector_scale(localInstance.n, &localInstance.x[0], (D) 0);
		localResiduals.resize(localInstance.m);
		globalResiduals.resize(globalInstance.m);
		residualsBuffer.resize(globalInstance.m);
		residual_updates_to_exchange.resize(globalInstance.m);

		omp_set_num_threads(distributedSettings->totalThreads);
		randomNumberUtil::init_omp_random_seeds();
		rs = randomNumberUtil::inittializeRandomSeeds(TOTAL_THREADS);
		randomNumberUtil::init_random_seeds(rs);

	}

	virtual void initializeAll() {
		this->initializeBuffers();
		this->initializeLossFunction();
		loss->computeReciprocalLipConstants();
		loss->recomputeResiduals();
	}

	D lastComputedObjectiveValue;
	virtual D getObjectiveValue() {
		lastComputedObjectiveValue = this->loss->computeObjectiveValue();
		return lastComputedObjectiveValue;
	}

	virtual std::string getResultLogHeaders() {
		return this->loss->getResultLogHeaders();
	}
	virtual std::string getResultLogRow(OptimizationStatistics& statistics) {
		return this->loss->getResultLogRow(statistics);
	}

	virtual void recomputeResiduals() {
		loss->recomputeResiduals();
		this->clearResidualBuffers();
	}

	virtual void executeBulkOfIterations() {
		L innerIterations = distributedSettings->iterationsPerThread
				* TOTAL_THREADS;

//		innerIterations = 1;
//		distributedSettings->iters_communicate_count = 100;
		for (unsigned int i = 0;
				i < distributedSettings->iters_communicate_count; i++) {

			cblas_set_to_zero(globalInstance.m, residual_updatesPtr);
//#pragma omp parallel
			{
#pragma omp parallel for
				for (int i = 0; i < innerIterations; i++) {
//				for (L i = 0; i < this->localInstance.n; i++) {
					L idx = gsl_rng_uniform_int(gsl_rng_r,
							this->localInstance.n);
//					L idx = i;

//				for (int i = 0; i < innerIterations; i++) {
//					L idx = gsl_rng_uniform_int(gsl_rng_r, localInstance.n);
					this->loss->performOneCoordinateUpdate(idx);
				}
			}
			this->exchange_residuals(residual_updates);
		}
	}

	virtual void executeBulkOfIterations2() {
		L innerIterations = distributedSettings->iterationsPerThread;
		/* =========================================MAIN LOOP =======================*/
//		shared(currentWorkerIteration)
#pragma omp parallel
		{
			if (my_thread_id == 0) {
				for (L comm_iteration = 0;
						comm_iteration
								< distributedSettings->iters_communicate_count;
						comm_iteration++) {
#pragma omp barrier
//					if (settings.distributed
//							== AsynchronousStreamlinedOptimized) {
//						resudial_updates_ptr =
//								&past_update[AsynchronousStreamlinedOptimizedUpdate
//										* part_global.m];
//					}
					for (L i = 0; i < globalResiduals.size(); i++) {
						D tmpVal = residual_updatesPtr[i];
						residual_updates_to_exchange[i] = tmpVal;
						globalResiduals[i] += tmpVal;
						parallel::atomic_add(residual_updatesPtr[i], -tmpVal);

					}
					this->exchange_residuals(residual_updates_to_exchange);
				}
			} else {
				for (L comm_iteration = 0;
						comm_iteration
								< distributedSettings->iters_communicate_count;
						comm_iteration++) {
					for (int i = 0; i < innerIterations; i++) {
						L idx = gsl_rng_uniform_int(gsl_rng_r, localInstance.n);
						this->loss->performOneCoordinateUpdate(idx);
					}
#pragma omp barrier
				}
			}
		}

		/* =========================================MAIN LOOP - ENDS =======================*/

	}

	~ClusterEngineExecutor() {
//		free(this->loss);
	}

	std::list<std::vector<D> > buffer;
	std::vector<D> past_update;
	std::vector<D> residual_updates_tosum;
	std::vector<D> exchanged;
	int exchange_data;

	virtual void initializeBuffers() {

		residual_updates.resize(globalInstance.m, 0);
		residual_updatesPtr = &residual_updates[0];

		if (distributedSettings->distributed == SynchronousReduce) {
			exchanged.resize(globalInstance.m);
		}

//
//		int buffer_size = world.size() - 1;
//		if (settings.distributed == AsynchronousBuffered) {
//			std::vector < D > empty_v(part.m, 0);
//			buffer.resize(buffer_size, empty_v);
//		}
//
//		if (settings.distributed == AsynchronousStreamlined) {
//			past_update.resize(part.m, 0);
//			buffer.resize(buffer_size, past_update);
//		}
//
//		if (settings.distributed == AsynchronousStreamlinedV2) {
//			past_update.resize(part.m, 0);
//			buffer.resize(buffer_size, past_update);
//		}
//
//		if (settings.distributed == AsynchronousTorus) {
//			past_update.resize(part.m, 0);
//			buffer_size = world.size() / settings.torus_width;
//			buffer.resize(buffer_size, past_update);
//			residual_updates_tosum.resize(part.m * (settings.torus_width - 1),
//					0);
//			exchanged.resize(part.m, 0);
//		}
//
//		if (settings.distributed == AsynchronousTorusOpt
//				|| settings.distributed == AsynchronousTorusOptCollectives) {
//
//			int rung_root = settings.topology.this_rung_index(world.size(),
//					world.rank(), settings.torus_width, 0);
//			if (rung_root == world.rank()) {
//				settings.topology.local_rung_communicator = world.split(
//						rung_root, 0);
//			} else {
//				settings.topology.local_rung_communicator = world.split(
//						rung_root);
//			}
//			past_update.resize(part.m, 0);
//			buffer_size = world.size() / settings.torus_width;
//			buffer.resize(buffer_size, past_update);
//			residual_updates_tosum.resize(part.m * (settings.torus_width - 1),
//					0);
//			exchanged.resize(part.m, 0);
//		}
//
//		if (settings.distributed == AsynchronousTorusCollectives) {
//			past_update.resize(part.m, 0);
//			buffer_size = world.size() / settings.torus_width;
//			buffer.resize(buffer_size, past_update);
//			torus_collectives_prepare(env, world, rung, settings);
//		}
//
		if (distributedSettings->distributed
				== AsynchronousStreamlinedOptimized) {
			past_update.resize(globalInstance.m * (world.size()), 0);
			exchanged.resize(globalInstance.m * 2, 0);
			exchange_data = 0;

		}
	}

	virtual inline void clearResidualBuffers() {
		cblas_set_to_zero(globalInstance.m, residual_updatesPtr);

		if (distributedSettings->distributed
				== AsynchronousStreamlinedOptimized) {
			cblas_set_to_zero(exchanged);
			cblas_set_to_zero(past_update);
		}

	}

	virtual inline void exchange_residuals(
			std::vector<D> & residual_updates_to_exchange
//			residuals,
//			std::vector<D> &residual_updates,
//			ProblemData<L, D> &part,
//			data_distributor<L, D> &dataDistributor,
//			distributed_statistics &stat,
//			std::list<std::vector<D> > &buffer,
//			std::vector<D> &past_update,
//			DistributedSettings &settings,
//			mpi::communicator &rung,
//			std::vector<D> &exchanged,
//			int &exchange_data,
//			int &AsynchronousStreamlinedOptimizedUpdate,
//			std::vector<D> &residual_updates_tosum
			) {

		switch (this->distributedSettings->distributed) {
		case SynchronousReduce:
			reduce_residuals(this->world, this->globalResiduals,
					residual_updates_to_exchange, this->residualsBuffer);
			break;
//		case SynchronousGather:
//			gather_residuals(world, residuals, residual_updates, part);
//			break;
//		case SynchronousPointToPoint:
//		case SynchronousSparse:
//			// NOTE: shift_residuals_point_to_point now extends shift_residuals_sparse
//			shift_residuals_point_to_point(env, world, dataDistributor,
//					residuals, residual_updates, stat.time_rounds);
//			break;
//
//		case SynchronousSupersparse:
//			// FIXME: Supersparse should be fixed
//			shift_residuals_supersparse(env, world, dataDistributor, residuals,
//					residual_updates, settings, stat.time_rounds);
//			break;
//
//		case AsynchronousStreamlined:
//			// NOTE: Token ring is a torus of width 1, so we could use that
//			shift_residuals_buffered_streamlined(env, world, buffer,
//					past_update, residuals, residual_updates, settings,
//					stat.time_rounds);
//			break;
//
//		case AsynchronousStreamlinedV2:
//			// NOTE: Token ring is a torus of width 1, so we could use that
//			shift_residuals_buffered_streamlined_v2(env, world, buffer,
//					past_update, residuals, residual_updates, settings,
//					stat.time_rounds);
//			break;
//
		case AsynchronousStreamlinedOptimized:
//			// NOTE: Token ring is a torus of width 1, so we could use that
			shift_residuals_buffered_streamlined_optimized(world,
					residual_updates_to_exchange, globalResiduals, past_update,
					exchanged, exchange_data,
					AsynchronousStreamlinedOptimizedUpdate);

			break;
//
//		case AsynchronousTorus:
//			shift_residuals_torus(env, world, residual_updates_tosum, exchanged,
//					buffer, past_update, residuals, residual_updates, settings,
//					stat.time_rounds);
//			break;
//
//		case AsynchronousTorusOpt: {
//			shift_residuals_torus_opt(env, world, residual_updates_tosum,
//					exchanged, buffer, past_update, residuals, residual_updates,
//					settings, stat.time_rounds,
//					settings.topology.local_rung_communicator);
//		}
//			break;
//		case AsynchronousTorusOptCollectives: {
//			shift_residuals_torus_opt_collectives(env, world,
//					residual_updates_tosum, exchanged, buffer, past_update,
//					residuals, residual_updates, settings, stat.time_rounds,
//					settings.topology.local_rung_communicator);
//		}
//			break;
//
//		case AsynchronousTorusCollectives:
//			shift_residuals_torus_collectives(env, world, rung, buffer,
//					past_update, residuals, residual_updates, settings,
//					stat.time_rounds);
//			break;
//
		}
	}

};

#endif /* CLUSTERENGINEEXECUTOR_H_ */
