/*
 * A matrix multiples a vector
 *
 * 
 */
#include <vector>
#include <iostream>
using namespace std;

#ifndef MATRIX_VECTOR_H
#define MATRIX_VECTOR_H


template<typename L, typename D>
void matrixvector(std::vector<D> &A, std::vector<L> &Asub, std::vector<L> &xA, std::vector<D> &b, L &m, std::vector<D> &result) {

	cblas_set_to_zero(result);

	for (L i = 0; i < m; i++){

		D temp = 0;
		L beginrow = xA[i];
		L endrow = xA[i+1] - 1;
		for (L j = beginrow; j <= endrow; j++)
			temp += A[j] * b[Asub[j]];
		result[i] = temp;
	}


}

template<typename L, typename D>
void vectormatrix( std::vector<D> &b, std::vector<D> &A, std::vector<L> &Asub, std::vector<L> &xA, L &m, std::vector<D> &result) {


	for (L i = 0; i < m; i++){
		L beginrow = xA[i];
		L endrow = xA[i+1] - 1;
		for (L j = beginrow; j <= endrow; j++)
			result[Asub[j]] += b[i] * A[j];
	}

}

template<typename L, typename D>
void matrixvector_b(std::vector<D> &A, std::vector<L> &Asub, std::vector<L> &xA, std::vector<D> &b,
		std::vector<D> &yy, D &scaler, L &m, std::vector<D> &result) {

	cblas_set_to_zero(result);

	for (L i = 0; i < m; i++){
		D temp = 0;
		L beginrow = xA[i];
		L endrow = xA[i+1] - 1;
		for (L j = beginrow; j <= endrow; j++)
			temp += scaler * A[j] * b[Asub[j]] *yy[Asub[j]] ;
		result[i] = temp;
	}


}

template<typename L, typename D>
void vectormatrix_b( std::vector<D> &b, std::vector<D> &A, std::vector<L> &Asub, std::vector<L> &xA,
		std::vector<D> &yy, D scaler, L &m, std::vector<D> &result) {

	cblas_set_to_zero(result);
	for (L i = 0; i < m; i++){
		L beginrow = xA[i];
		L endrow = xA[i+1] - 1;
//		cout<<endrow<<endl;
		//if (endrow>=1800000)
		//cout<<xA[i]<<"   "<<xA[i+1]-1<<"    "<<endrow<<endl;
		for (L j = beginrow; j <= endrow; j++){
			result[Asub[j]] += scaler * A[j] * b[i] * yy[i];
		}
	}
}

#endif /* MATRIX_VECTOR_H */
