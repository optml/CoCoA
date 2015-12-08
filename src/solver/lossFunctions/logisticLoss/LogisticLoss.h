/*
 * SquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef LogisticLOSS_H_
#define LogisticLOSS_H_

#include "../AbstractLoss.h"
#include "../../treshhold_functions.h"
namespace Loss {

template<typename L, typename D>
class LogisticLoss: public Loss::AbstractLoss<L, D> {
public:


	LogisticLoss() {

	}
	virtual ~LogisticLoss() {

	}

	D lastComputedObjectiveValue;
	D trainAccuracy;
	D testAccuracy;
	virtual std::string getResultLogHeaders() {
		return "Elapsed iterations, Current objective value, Pure computation time, Total time, Train 0-1 loss, Test 0-1 loss";
	}
	virtual std::string getResultLogRow(OptimizationStatistics& statistics) {
		std::stringstream ss;
		ss << statistics.elapsedIterations << "," << setprecision(16)
				<< lastComputedObjectiveValue << ","
				<< statistics.elapsedPureComputationTime << ","
				<< statistics.elapsed_time<<","<<this->trainAccuracy<<","<<this->testAccuracy;
		return ss.str();
	}

};

}
#endif /* SQUARELOSS_H_ */
