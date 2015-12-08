/*
 * SquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef SVMDualLoss_H_
#define SVMDualLoss_H_

#include "../AbstractLoss.h"

namespace Loss {

template<typename L, typename D>
class SVMDualLoss: public Loss::AbstractLoss<L, D> {
public:
	SVMDualLoss() {

	}
	virtual ~SVMDualLoss() {

	}

	D lastPrimalObjectiveValue;
	D lastDualObjectiveValue;
	D lastZeroOneAccuracy;
	D lastTestZeroOneAccuracy;
	D lastComputedObjectiveValue;

	virtual std::string getResultLogHeaders() {
		return "Elapsed iterations, Current duality gap, Primal objective, Dual objective, 0-1 loss, Pure computation time, Total time, Test 0-1";
	}
	virtual std::string getResultLogRow(OptimizationStatistics& statistics) {
		std::stringstream ss;
		ss << statistics.elapsedIterations << "," << setprecision(16)
				<< lastComputedObjectiveValue << "," << lastPrimalObjectiveValue
				<< "," << lastDualObjectiveValue << "," << lastZeroOneAccuracy
				<< "," << statistics.elapsedPureComputationTime << ","
				<< statistics.elapsed_time<<","<<lastTestZeroOneAccuracy;
		return ss.str();
	}

};

}
#endif /* SQUARELOSS_H_ */
