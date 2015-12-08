/*
 * Context.h
 *
 *  Created on: Sep 9, 2013
 *      Author: taki
 */

#ifndef CONTEXT_H_
#define CONTEXT_H_
#include <string>
#include "solver/settingsAndStatistics.h"
class Context {
public:
	std::string resultFile;
	bool dataANSIInput;
	std::string matrixAFile;
	std::string matrixATestFile;
	std::string vectorBFile;

	std::string experimentName;

	OptimizationSettings& settings;
	int xDim;
	int yDim;
	int zDim;
	bool isTestErrorFileAvailable;
	double lambda;
	double tmp;
	int experiment;
	int numberOfThreads;
	Context(OptimizationSettings& _settings) :
			settings(_settings) {
		settings.verbose = true;
		dataANSIInput=true;
		zDim = 0;
		isTestErrorFileAvailable=false;
		lambda = 0;
		tmp=0;
		numberOfThreads = 1;
	}

};

#endif /* CONTEXT_H_ */
