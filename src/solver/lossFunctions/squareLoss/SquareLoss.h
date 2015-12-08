/*
 * SquareLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef SQUARELOSS_H_
#define SQUARELOSS_H_

#include "../AbstractLoss.h"
#include "../../treshhold_functions.h"
namespace Loss {

template<typename L, typename D>
class SquareLoss: public Loss::AbstractLoss<L, D> {
public:


	SquareLoss() {

	}
	virtual ~SquareLoss() {

	}

	D lastComputedObjectiveValue;
	virtual std::string getResultLogHeaders() {
		return "Elapsed iterations, Current objective value, Pure computation time, Total time";
	}
	virtual std::string getResultLogRow(OptimizationStatistics& statistics) {
		std::stringstream ss;
		ss << statistics.elapsedIterations << "," << setprecision(16)
				<< lastComputedObjectiveValue << ","
				<< statistics.elapsedPureComputationTime << ","
				<< statistics.elapsed_time;
		return ss.str();
	}

};

}
#endif /* SQUARELOSS_H_ */
