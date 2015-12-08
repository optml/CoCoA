/*
 * MulticoreEngineExecutor.h
 *
 *  Created on: Sep 10, 2013
 *      Author: taki
 */

#ifndef MULTICOREENGINEEXECUTOR_H_
#define MULTICOREENGINEEXECUTOR_H_
#include "AbstractEngineExecutor.h"
#include "./structures.h"
#include "settingsAndStatistics.h"

#include "lossFunctions/multicoreLossFunctions.h"
#include "../utils/randomNumbersUtil.h"
template<typename L, typename D>
class MulticoreEngineExecutor: public AbstractEngineExecutor<L, D> {

public:

	ProblemData<L, D> & instance;
	Loss::AbstractLoss<L, D>* loss;
	std::vector<D> residuals;
	std::vector<gsl_rng *> rs;
	MulticoreEngineExecutor(ProblemData<L, D> & _instance,
			OptimizationSettings* _settings) :
			instance(_instance) {
		this->settings = _settings;
		instance.x.resize(instance.n);
		cblas_vector_scale(instance.n, &instance.x[0], (D) 0);
		residuals.resize(instance.m);
		switch (_settings->lossFunction) {
		case 0:
			this->loss = new Loss::MulticoreL1RegSquareLoss<L, D>(&instance,
					&residuals);
			break;
		case 1:
			this->loss = new Loss::MulticoreL2SquareRegSquareLoss<L, D>(
					&instance, &residuals);
			break;
		case 2:
			this->loss = new Loss::MulticoreSVMDualLoss<L, D>(instance,
					residuals);
			break;

		case 3:
			this->loss = new Loss::MulticoreL2SquareRegLogisticLoss<L, D>(
					&instance, &residuals);

			break;

		case 4:

			break;

		default:
			break;
		}
		loss->recomputeResiduals();
		loss->computeReciprocalLipConstants();

		omp_set_num_threads(this->settings->totalThreads);
		randomNumberUtil::init_omp_random_seeds();
		rs = randomNumberUtil::inittializeRandomSeeds(TOTAL_THREADS);
		randomNumberUtil::init_random_seeds(rs);
	}

	D getObjectiveValue() {
		return this->loss->computeObjectiveValue();
	}

	virtual std::string getResultLogHeaders() {
		return loss->getResultLogHeaders();
	}
	virtual std::string getResultLogRow(OptimizationStatistics& statistics) {
		return loss->getResultLogRow(statistics);
	}

	void recomputeResiduals() {
		loss->recomputeResiduals();
	}

	void executeBulkOfIterations() {
#pragma omp parallel for
		for (unsigned int i = 0; i < this->settings->innerIterations; i++) {
			L idx = gsl_rng_uniform_int(gsl_rng_r, instance.n);
			this->loss->performOneCoordinateUpdate(idx);
		}
	}





	~MulticoreEngineExecutor() {
//		free(this->loss);
	}

};

#endif /* MULTICOREENGINEEXECUTOR_H_ */
