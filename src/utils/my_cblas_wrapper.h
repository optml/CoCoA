/*
 * my_cblas_wrapper.h
 *
 *  Created on: Apr 14, 2012
 *      Author: taki
 */

#ifndef MY_CBLAS_WRAPPER_H_
#define MY_CBLAS_WRAPPER_H_
#include <vector>
#include <gsl/gsl_cblas.h>

template<typename F>
void cblas_vector_scale(const int n, F* vector, const F factor) {
}

template<>
void cblas_vector_scale(const int n, double* vector, const double factor) {
	cblas_dscal(n, factor, vector, 1);
}

template<>
void cblas_vector_scale(const int n, float* vector, const float factor) {
	cblas_sscal(n, factor, vector, 1);
}

template<typename F>
void cblas_set_to_zero(std::vector<F> &vector) {
	const F zero = 0;
	cblas_vector_scale(vector.size(), &vector[0], zero);
}

template<typename F>
void cblas_set_to_zero(const int n, F* vector) {
	const F zero = 0;
	cblas_vector_scale(n, vector, zero);
}



template<typename F>
void cblas_sum_of_vectors(std::vector<F> &result, std::vector<F> &to_add, F to_add_scale) {
}

template<>
void cblas_sum_of_vectors(std::vector<double> &result,
		std::vector<double> &to_add,double to_add_scale) {
	cblas_daxpy(result.size(), to_add_scale, &to_add[0], 1, &result[0], 1);
}
template<>
void cblas_sum_of_vectors(std::vector<float> &result,
		std::vector<float> &to_add, float to_add_scale) {
	cblas_saxpy(result.size(), to_add_scale, &to_add[0], 1, &result[0], 1);
}






template<typename F>
void cblas_sum_of_vectors(std::vector<F> &result, std::vector<F> &to_add) {
}

template<>
void cblas_sum_of_vectors(std::vector<double> &result,
		std::vector<double> &to_add) {
	double one = 1;
	cblas_daxpy(result.size(), one, &to_add[0], 1, &result[0], 1);
}
template<>
void cblas_sum_of_vectors(std::vector<float> &result,
		std::vector<float> &to_add) {
	float one = 1;
	cblas_saxpy(result.size(), one, &to_add[0], 1, &result[0], 1);
}

template<typename F>
void cblas_sum_of_vectors(const unsigned int N, F* result, F* to_add) {
}

template<>
void cblas_sum_of_vectors(const unsigned int N, double* result,
		double* to_add) {
	double one = 1;
	cblas_daxpy(N, one, to_add, 1, result, 1);
}
template<>
void cblas_sum_of_vectors(const unsigned int N, float* result, float* to_add) {
	float one = 1;
	cblas_saxpy(N, one, to_add, 1, result, 1);
}

template<typename F>
void cblas_subtract_of_vectors(const unsigned int N, F* result, F* to_add) {
}

template<>
void cblas_subtract_of_vectors(const unsigned int N, double* result,
		double* to_add) {
	double minusone = -1;
	cblas_daxpy(N, minusone, to_add, 1, result, 1);
}
template<>
void cblas_subtract_of_vectors(const unsigned int N, float* result,
		float* to_add) {
	float minusone = -1;
	cblas_saxpy(N, minusone, to_add, 1, result, 1);
}

//template<typename F>
//void cblas_matrix_matrix_multiply(const CBLAS_ORDER Order,
//		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
//		const int M, const int N, const int K, const F alpha,
//		const F *A, const int lda, const F *B, const int ldb,
//		const F beta, F *C, const int ldc) {
//}

//template<>
void cblas_matrix_matrix_multiply(const CBLAS_ORDER Order,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
		const int N, const int K, const double alpha, const double *A,
		const int lda, const double *B, const int ldb, const double beta,
		double *C, const int ldc) {
	cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
			ldc);
}

//template<>
void cblas_matrix_matrix_multiply(const CBLAS_ORDER Order,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M,
		const int N, const int K, const float alpha, const float *A,
		const int lda, const float *B, const int ldb, const float beta,
		float *C, const int ldc) {
	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
			ldc);
}

double cblas_l1_norm(const int N, const double *X, const int incX) {
	return cblas_dasum(N, X, incX);
}

float cblas_l1_norm(const int N, const float *X, const int incX) {
	return cblas_sasum(N, X, incX);
}

double cblas_l2_norm(const int N, const double *X, const int incX) {
	return cblas_dnrm2(N, X, incX);
}

float cblas_l2_norm(const int N, const float *X, const int incX) {
	return cblas_snrm2(N, X, incX);
}

template<typename F>
void cblas_vector_copy(std::vector<F> &from, std::vector<F> &to) {
}

template<>
void cblas_vector_copy(std::vector<double> &from, std::vector<double> &to) {
	cblas_dcopy(from.size(), &from[0], 1, &to[0], 1);
}

template<>
void cblas_vector_copy(std::vector<float> &from, std::vector<float> &to) {
	cblas_scopy(from.size(), &from[0], 1, &to[0], 1);
}

void cblas_vector_copy(const int N, const double *X, const int incX, double *Y,
		const int incY) {
	cblas_dcopy(N, X, incX, Y, incY);
}

void cblas_vector_copy(const int N, const float *X, const int incX, float *Y,
		const int incY) {
	cblas_scopy(N, X, incX, Y, incY);
}

CBLAS_INDEX cblas_vector_max_index(const int N, const double *X,
		const int incX) {
	return cblas_idamax(N, X, incX);
}

CBLAS_INDEX cblas_vector_max_index(const int N, const float *X,
		const int incX) {
	return cblas_isamax(N, X, incX);
}

#endif /* MY_CBLAS_WRAPPER_H_ */
