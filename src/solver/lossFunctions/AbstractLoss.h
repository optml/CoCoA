/*
 * AbstractLoss.h
 *
 *  Created on: Sep 11, 2013
 *      Author: taki
 */

#ifndef ABSTRACTLOSS_H_
#define ABSTRACTLOSS_H_
#include "../structures.h"
#include <vector>
namespace Loss {

template<typename L, typename D>
class AbstractLoss {
public:
	AbstractLoss() {

	}
	virtual ~AbstractLoss() {

	}

	virtual std::string getResultLogHeaders() {
		return "Function is not implemented";
	}
	virtual std::string getResultLogRow(OptimizationStatistics& statistics) {
		return "Function is not implemented";
	}

	virtual void recomputeResiduals() {

	}

	virtual void performOneCoordinateUpdate(L idx) {

	}

	virtual void computeReciprocalLipConstants() {

	}

	virtual D computeObjectiveValue() {
		return 0;
	}

};

}
#endif /* ABSTRACTLOSS_H_ */
