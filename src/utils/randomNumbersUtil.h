/*
 * randomNumbersUtil.h
 *
 *  Created on: Jul 2, 2013
 *      Author: taki
 */

#ifndef RANDOMNUMBERSUTIL_H_
#define RANDOMNUMBERSUTIL_H_

#include <gsl/gsl_rng.h>
const gsl_rng_type * gsl_rng_T;
gsl_rng * gsl_rng_r;
#pragma omp threadprivate (gsl_rng_r)
#pragma omp threadprivate (gsl_rng_T)

#include <omp.h>
unsigned int myseed = 0;
unsigned int myseed2 = 0;
unsigned int my_thread_id = 0;
unsigned int TOTAL_THREADS = 1;
#pragma omp threadprivate (myseed)
#pragma omp threadprivate (myseed2)
#pragma omp threadprivate (my_thread_id)


namespace randomNumberUtil {

void init_omp_random_seeds() {
	int initial_seed = 0;
	TOTAL_THREADS = 1;
#pragma omp parallel
	{
		TOTAL_THREADS = omp_get_num_threads();
	}
	unsigned int seed[TOTAL_THREADS];
	unsigned int seed2[TOTAL_THREADS];
#pragma omp parallel
	{
		my_thread_id = omp_get_thread_num();
		if (omp_get_thread_num() == 0) {
			srand(initial_seed);
			for (unsigned int i = 0; i < TOTAL_THREADS; i++) {
				seed[i] = (unsigned int) RAND_MAX * rand();
				seed2[i] = (unsigned int) RAND_MAX * rand();
			}
		}
	}
#pragma omp parallel
	{
		myseed = seed[omp_get_thread_num()];
		myseed2 = seed2[omp_get_thread_num()];
	}
}

template<typename L>
void init_omp_random_seeds(L initial_seed) {
	TOTAL_THREADS = 1;
#pragma omp parallel
	{
		TOTAL_THREADS = omp_get_num_threads();
	}
	unsigned int seed[TOTAL_THREADS];
	unsigned int seed2[TOTAL_THREADS];
#pragma omp parallel
	{
		my_thread_id = omp_get_thread_num();
		if (omp_get_thread_num() == 0) {
			srand(initial_seed);
			for (unsigned int i = 0; i < TOTAL_THREADS; i++) {
				seed[i] = (unsigned int) RAND_MAX * rand();
				seed2[i] = (unsigned int) RAND_MAX * rand();
			}
		}
	}
#pragma omp parallel
	{
		myseed = seed[omp_get_thread_num()];
		myseed2 = seed2[omp_get_thread_num()];
	}
}

void init_random_seeds(std::vector<gsl_rng *>& rs, int offset = 0) {
#pragma omp parallel
	{
		my_thread_id = omp_get_thread_num();
		gsl_rng_r = rs[my_thread_id];
	}
	TOTAL_THREADS = 1;
#pragma omp parallel
	{
		TOTAL_THREADS = omp_get_num_threads();
	}
	cout << "Using " << TOTAL_THREADS << " threads " << endl;
	unsigned int seed[TOTAL_THREADS];
#pragma omp parallel
	{
		gsl_rng_set(gsl_rng_r, my_thread_id + offset);
		if (omp_get_thread_num() == 0) {

			for (unsigned int i = 0; i < TOTAL_THREADS; i++)
				seed[i] = gsl_rng_uniform_int(gsl_rng_r, RAND_MAX);
		}
	}
#pragma omp parallel
	{
		myseed = seed[omp_get_thread_num()];
	}
}

std::vector<gsl_rng *> inittializeRandomSeeds(const int MAXIMUM_THREADS = 24) {
// setup GSL random generators
	gsl_rng_env_setup();
	const gsl_rng_type * T;
	gsl_rng * r;
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	std::vector<gsl_rng *> rs(MAXIMUM_THREADS);
	for (int i = 0; i < MAXIMUM_THREADS; i++) {
		rs[i] = gsl_rng_alloc(T);
		gsl_rng_set(rs[i], i);
	}
//	init_omp_random_seeds();
	return rs;
}

}

#endif /* RANDOMNUMBERSUTIL_H_ */
