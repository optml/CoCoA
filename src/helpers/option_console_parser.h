/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 */

/*
 * option_parser.h
 *
 *  Created on: Sep 12, 2012
 *      Author: taki
 */

#ifndef OPTION_PARSER_H_
#define OPTION_PARSER_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include "Context.h"

namespace consoleHelper {

using namespace std;

void print_usage() {
	cout << "Usage:" << endl;
	cout << "-------------------------------------" << endl;
	cout << "Parameters:" << endl;

	cout << "-------------------------------------" << endl;

}

void setParameter(Context &context, char c ) {

	switch (c) {
	case 'x':
		context.xDim = atoi(optarg);
		break;
	case 'y':
		context.yDim = atoi(optarg);
		break;
	case 'z':
		context.zDim = atoi(optarg);
		break;
	case 'r':
		context.resultFile = optarg;
		break;
	case 'A':
		context.matrixAFile = optarg;
		break;
	case 'b':
		context.vectorBFile = optarg;
		break;
	case 'e':
		context.experiment = atoi(optarg);
		break;
	case 'E':
		context.experimentName=optarg;
		break;
	case 'l':
		context.lambda = atof(optarg);
		break;

	case 'T':
//		context.isTestErrorFileAvailable=true;
//		context.matrixATestFile=optarg;
		break;

	case 'c':
		context.settings.bulkIterations = atoi(optarg);
		break;
	case 't':
		context.settings.totalThreads = atoi(optarg);
		break;
	case 'f':
		context.settings.lossFunction = atoi(optarg);
		break;
	case 'I':
		context.settings.innerIterations = atoll(optarg);
	case 'M':
		context.settings.LocalMethods = atoll(optarg);
		break;
	case 'i':
		context.settings.showIntermediateObjectiveValue = atoi(optarg);
		context.settings.showInitialObjectiveValue = atoi(optarg);
		context.settings.showLastObjectiveValue = atoi(optarg);
		break;
	}
}

int parseConsoleOptions(Context &context, int argc, char *argv[]) {

	char c;
	/*
	 * r - result file
	 */

	while ((c = getopt(argc, argv, "x:y:z:r:e:A:b:l:t:c:f:i:I:T:")) != -1) {
		consoleHelper::setParameter(context, c);
	}

	return 0;
}
}
#endif /* OPTION_PARSER_H_ */
