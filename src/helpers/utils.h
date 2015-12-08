/*
 * This file contains an functions for experiment done in paper
 * P. Richtárik and M. Takáč
 *      Parallel Coordinate Descent Methods for Big Data Optimization
 * http://www.optimization-online.org/DB_HTML/2012/11/3688.html
 */


/*
 * utils.h
 *
 *  Created on: Nov 25, 2012
 *      Author: taki
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <sys/time.h>


double gettime_(void) {
	struct timeval timer;
	if (gettimeofday(&timer, NULL))
		return -1.0;
	return timer.tv_sec + 1.0e-6 * timer.tv_usec;
}

template <typename T> int sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}


#endif /* UTILS_H_ */
