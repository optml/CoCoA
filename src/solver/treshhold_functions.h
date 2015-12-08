/*
 * This file contains an functions for experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */



/*
 * Set of funcitons and structures
 */

#ifndef SOFT_TRESHHOLD_H_
#define SOFT_TRESHHOLD_H_

#include <math.h>
#include "../helpers/utils.h"

/*
 *       Softtresholding formula
 *        sgn(X) (|X|-T)_+
 *           X-T  X>T
 *           X+T  X<-T
 *            0    otherwise
 */
template<typename T>
T compute_soft_treshold(T valT, T valX) {
	T absXminusT = fabs(valX) - valT;
	if (absXminusT < 0)
		absXminusT = 0;
	return sgn(valX) * absXminusT;
}


#endif /* SOFT_TRESHHOLD_H_ */
