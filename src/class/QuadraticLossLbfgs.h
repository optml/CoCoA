/*
 * QuadraticLossCD.h
 *
 *  Created on: Dec 1, 2014
 *      Author: taki
 */

#ifndef QuadraticLossLbfgs_H_
#define QuadraticLossLbfgs_H_
#include "QuadraticLoss.h"

#include "engine.h"
#define  BUFSIZE 256

template<typename L, typename D>
class QuadraticLossLbfgs: public QuadraticLoss<L, D> {
public:

	Engine *ep;
	mxArray *T = NULL, *result = NULL;
	char buffer[BUFSIZE + 1];

	QuadraticLossLbfgs() {

	}

	mxArray * mRow = NULL, *mCol = NULL, *mVal = NULL, *mAlpha = NULL, *mW =
			NULL, *lambda = NULL, *totalN = NULL, *alphaInit = NULL, *my = NULL;

	virtual void init(ProblemData<L, D> &instance) {
		if (!(ep = engOpen("\0"))) {
			fprintf(stderr, "\nCan't start MATLAB engine\n");
			exit(1);
//				return EXIT_FAILURE;
		}

		L nnz = instance.A_csr_col_idx.size();

		std::vector<double> ROW(nnz);
		std::vector<double> COL(nnz);
		std::vector<double> VAL(nnz);

		for (L i = 0; i < instance.n; i++) {
			for (L j = instance.A_csr_row_ptr[i];
					j < instance.A_csr_row_ptr[i + 1]; j++) {
				ROW[j] = i + 1;
				COL[j] = instance.A_csr_col_idx[j] + 1;
				VAL[j] = instance.A_csr_values[j];
			}
		}
		mRow = mxCreateDoubleMatrix(1, nnz, mxREAL);
		memcpy((void *) mxGetPr(mRow), (void *) &COL[0], sizeof(double) * nnz);
		mCol = mxCreateDoubleMatrix(1, nnz, mxREAL);
		memcpy((void *) mxGetPr(mCol), (void *) &ROW[0], sizeof(double) * nnz);
		mVal = mxCreateDoubleMatrix(1, nnz, mxREAL);
		memcpy((void *) mxGetPr(mVal), (void *) &VAL[0], sizeof(double) * nnz);

		ROW.resize(0);
		COL.resize(0);
		VAL.resize(0);

		my = mxCreateDoubleMatrix(instance.n, 1, mxREAL);
		std::vector<double> Y(instance.n);
		for (L i = 0; i < instance.n; i++) {
			Y[i] = instance.b[i];
		}
		memcpy((void *) mxGetPr(my), (void *) &Y[0],
				sizeof(double) * instance.n);
		Y.resize(0);
		engPutVariable(ep, "my", my);

		mAlpha = mxCreateDoubleMatrix(instance.n, 1, mxREAL);
		memcpy((void *) mxGetPr(mAlpha), (void *) &instance.x[0],
				sizeof(double) * instance.n);

		for (L i = 0; i < instance.n; i++) {
			Y[i] = 0;
		}
		alphaInit = mxCreateDoubleMatrix(instance.n, 1, mxREAL);
		memcpy((void *) mxGetPr(alphaInit), (void *) &Y[0],
				sizeof(double) * instance.n);
		Y.resize(0);

		mW = mxCreateDoubleMatrix(instance.m, 1, mxREAL);

		engPutVariable(ep, "mRow", mRow);
		engPutVariable(ep, "mCol", mCol);
		engPutVariable(ep, "mVal", mVal);
		engPutVariable(ep, "mAlpha", mAlpha);
		engPutVariable(ep, "alphaInit", alphaInit);

		lambda = mxCreateDoubleMatrix(1, 1, mxREAL);
		memcpy((void *) mxGetPr(lambda), (void *) &instance.lambda,
				sizeof(double) * 1);
		double totalND = instance.total_n;
		totalN = mxCreateDoubleMatrix(1, 1, mxREAL);
		memcpy((void *) mxGetPr(totalN), (void *) &totalND, sizeof(double) * 1);

		engPutVariable(ep, "lambda", lambda);
		engPutVariable(ep, "totalN", totalN);

		engPutVariable(ep, "mW", mW);

		buffer[BUFSIZE] = '\0';
		engOutputBuffer(ep, buffer, BUFSIZE);

		engEvalString(ep, "  addpath('../libs/minFunc/minFunc') ");
		engEvalString(ep, "  addpath('../libs/cocoaLossFunctions') ");

		engEvalString(ep, " A=sparse(mRow, mCol, mVal)'; ");
		engEvalString(ep, " clear mRow; ");
		engEvalString(ep, " clear mCol; ");
		engEvalString(ep, " clear mVal; ");

		std::stringstream ss;
		ss << "options.Method = '" << instance.experimentName << "' ";
		engEvalString(ep, ss.str().c_str());
		ss.str("");
		ss << "options.optTol =  0.0000000000000001 ";
		engEvalString(ep, ss.str().c_str());
		ss.str("");
		ss << "options.progTol =  0.0000000000000001 ";
		engEvalString(ep, ss.str().c_str());

		ss.str("");
		ss << "options.MaxIter = " << instance.theta << " ";
		engEvalString(ep, ss.str().c_str());

	}

	virtual ~QuadraticLossLbfgs() {
		engClose(ep);
	}

	virtual void solveLocalProblem(ProblemData<L, D> &instance,
			std::vector<D> &deltaAlpha, std::vector<D> &w,
			std::vector<D> &deltaW, DistributedSettings & distributedSettings) {

		memcpy((void *) mxGetPr(mAlpha), (void *) &instance.x[0],
				sizeof(double) * instance.n);
		memcpy((void *) mxGetPr(mW), (void *) &w[0],
				sizeof(double) * instance.m);
		engPutVariable(ep, "mAlpha", mAlpha);
		engPutVariable(ep, "mW", mW);

		engEvalString(ep, "alphaInit=alphaInit*0; ");

		engEvalString(ep,
				"alphaInit =  minFunc(@SquareLoss,alphaInit,options,A,my,mAlpha,mW,lambda,totalN); ");
//		engEvalString(ep, "startVal - endVal;  ");

//		engEvalString(ep,
//				"alphaInit =  fminunc(@SquareLoss,alphaInit,options,A,my,mAlpha,mW,lambda,totalN ); ");

//		printf("%s", buffer);
//		printf("\nRetrieving X...\n");
		if ((result = engGetVariable(ep, "alphaInit")) == NULL)
			printf("Oops! You didn't create a variable alphaInit.\n\n");
		else {
//			printf("X is class %s\t\n", mxGetClassName(result));
		}
		memcpy((void *) &deltaAlpha[0], (void *) mxGetPr(result),
				sizeof(double) * instance.n);
		mxDestroyArray(result);

		for (L idx = 0; idx < instance.n; idx++) {

			for (L i = instance.A_csr_row_ptr[idx];
					i < instance.A_csr_row_ptr[idx + 1]; i++) {

				deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
						* instance.A_csr_values[i] * deltaAlpha[idx];

			}

		}

//

//		for (unsigned int it = 0; it < distributedSettings.iterationsPerThread;
//				it++) {
//
//			L idx = rand() / (0.0 + RAND_MAX) * instance.n;
//
//			// compute "delta alpha" = argmin
//
//			D dotProduct = 0;
//			for (L i = instance.A_csr_row_ptr[idx];
//					i < instance.A_csr_row_ptr[idx + 1]; i++) {
//
//				dotProduct += (w[instance.A_csr_col_idx[i]]
//						+ deltaW[instance.A_csr_col_idx[i]])
//						* instance.A_csr_values[i];
//
//			}
//
//			D alphaI = instance.x[idx] + deltaAlpha[idx];
//
//			D deltaAl = 0; // FINISH
//
//			D norm = cblas_l2_norm(
//					instance.A_csr_row_ptr[idx + 1]
//							- instance.A_csr_row_ptr[idx],
//					&instance.A_csr_values[instance.A_csr_row_ptr[idx]], 1);
//
//			D denom = norm * norm / instance.lambda / instance.n + 0.5;
//
//			deltaAl = (1.0 - 0.5 * alphaI - instance.b[idx] * dotProduct)
//					/ denom;
//
//			deltaAlpha[idx] += deltaAl;
//			for (L i = instance.A_csr_row_ptr[idx];
//					i < instance.A_csr_row_ptr[idx + 1]; i++) {
//
//				deltaW[instance.A_csr_col_idx[i]] += instance.oneOverLambdaN
//						* instance.A_csr_values[i] * deltaAl * instance.b[idx];
//
//			}
//
//		}
	}
};

#endif /* QUADRATICLOSSCD_H_ */
