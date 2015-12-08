/*
 * AbstractEngineExecutor.h
 *
 *  Created on: Sep 10, 2013
 *      Author: taki
 */

#ifndef ABSTRACTENGINEEXECUTOR_H_
#define ABSTRACTENGINEEXECUTOR_H_
#include "settingsAndStatistics.h"
#include "../utils/my_cblas_wrapper.h"
#include "../parallel/parallel_essentials.h"
template<typename L, typename D>
class AbstractEngineExecutor {
public:
	OptimizationSettings* settings;

	AbstractEngineExecutor() {
		this->settings = NULL;
	}

	virtual std::string getResultLogHeaders() {
		return "Function is not implemented";
	}
	virtual std::string getResultLogRow(OptimizationStatistics& statistics) {
		return "Function is not implemented";
	}

	virtual void executeBulkOfIterations() {
	}

	virtual void recomputeResiduals() {
	}

	virtual D getObjectiveValue() {
		return 0;
	}

};

#endif /* ABSTRACTENGINEEXECUTOR_H_ */
