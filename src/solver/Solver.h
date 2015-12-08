/*
 * Solver.h
 *
 *  Created on: Sep 10, 2013
 *      Author: taki
 */

#ifndef SOLVER_H_
#define SOLVER_H_

#include "AbstractEngineExecutor.h"
#include "settingsAndStatistics.h"
template<typename L, typename D>
class Solver {
public:
	AbstractEngineExecutor<L, D>& executor;
	OptimizationSettings *settings;
	Solver(AbstractEngineExecutor<L, D>& _executor) :
			executor(_executor) {
		settings = _executor.settings;
	}

	void runSolver() {
		if (settings->verbose)
			std::cout << "Number of large iterations: "
					<< settings->iters_bulkIterations_count << std::endl;
		D objVal;
		OptimizationStatistics statistics;


		if (settings->showInitialObjectiveValue) {
			objVal = executor.getObjectiveValue();
			if (settings->verbose) {
				std::cout << executor.getResultLogHeaders() << std::endl;
				std::string logRow = executor.getResultLogRow(statistics);
				std::cout <<logRow  << std::endl;
				if (settings->logToFile){
					(*settings->logFile)<<logRow  << std::endl;
				}
			}
		}

		double totalStart = gettime_();
		for (int i = 0; i < settings->iters_bulkIterations_count; i++) {
			double bulkStart = gettime_();
			executor.executeBulkOfIterations();
			double bulkEnd = gettime_();
			statistics.elapsedPureComputationTime += (bulkEnd - bulkStart);
			statistics.elapsed_time = bulkEnd - totalStart;
			statistics.elapsedIterations++;
			if (settings->recomputeResidualAfterEachBulkIteration) {
				executor.recomputeResiduals();
			}

			if (settings->showIntermediateObjectiveValue) {
				objVal = executor.getObjectiveValue();
				if (settings->verbose) {
					std::string logRow = executor.getResultLogRow(statistics);
					std::cout <<logRow  << std::endl;
					if (settings->logToFile){
						(*settings->logFile)<<logRow  << std::endl;
					}
				}
			}
		}




		if (settings->showLastObjectiveValue) {
			objVal = executor.getObjectiveValue();
			if (settings->verbose)
				std::cout << "Last objective value: " << setprecision(16)
						<< objVal << std::endl;
		}

	}

};

#endif /* SOLVER_H_ */
