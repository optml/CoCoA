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

#ifndef OPTIONDISTRIBUTED_PARSER_H_
#define OPTIONDISTRIBUTED_PARSER_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include "../Context.h"

namespace consoleHelper {

using namespace std;

int parseDistributedOptions(Context& context, DistributedSettings &settings,
		int argc, char *argv[]) {

	char c;
	/*
	 * r - result file
	 */
	int residualShif;
	while ((c = getopt(argc, argv, "x:y:z:r:e:A:b:l:t:c:f:g:i:I:T:S:E:C:a:B:F:p:M:"))
			!= -1) {
		switch (c) {
		case 'M':
			context.settings.LocalMethods = atoll(optarg);
			break;
		case 'F':
			context.dataANSIInput=atoi(optarg);
			break;

		case 'g':
			settings.forcedSigma = atof(optarg);
			break;

		case 'B':
			settings.iters_bulkIterations_count = atol(optarg);
			break;

		case 'T':
			settings.iterationsPerThread = atol(optarg);
			break;

		case 'a':
			settings.APPROX = atoi(optarg);
			break;
		case 'I':
			settings.iterationsPerThread = atoll(optarg);
			break;

		case 'p':
			context.tmp = atof(optarg);
			break;

		case 'C':
			settings.iters_communicate_count = atol(optarg);
			break;
		case 'S':
			residualShif = atoi(optarg);
			switch (residualShif) {
			case 0:
				settings.distributed = SynchronousReduce;
				break;
			case 1:
				settings.distributed = AsynchronousStreamlinedOptimized;
				break;
			}

			break;

		default:
			consoleHelper::setParameter(context, c);
			break;
		}

	}

	return 0;
}
}
#endif /* OPTION_PARSER_H_ */
