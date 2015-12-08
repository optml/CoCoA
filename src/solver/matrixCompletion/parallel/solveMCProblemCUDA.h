/*
 *  Paralle Matrix Completion Solver on GPU
 *
 * m - number of rows
 * n - number of cols
 * r - rank of matrix which we want to find
 * h_Z_values - known values
 * h_Z_col_idx - column index of known data
 * h_Z_row_idx - row index of known data
 * (h_Z_.....) is a sparse matrix in COO format
 * completed_matrix - is reconstructed matrix saved in
 *                    row oriented format
 * settings - setting to optimizer
 * mu - penalization parameter mu
 */

using namespace std;

#include <vector>
#include "../../structures.h"
#include "../../../helpers/matrix_conversions.h"
#include "../../../helpers/time_functions.h"
#include "../../../mc/mc_problem_generation.h"
#include "../../../headers/thrust_headers.h"
#include "../../../headers/cuda_headers.h"

void run_GPU_MC_Solver(int m, int n, int r, std::vector<float> h_Z_values,
		std::vector<int> h_Z_col_idx, std::vector<int> h_Z_row_idx,
		std::vector<float>* completed_matrix, OptimizationSettings settings,
		float mu) {
	std::vector<float> Z_csr_val, Z_csc_val;
	std::vector<int> Z_csr_colIdx, Z_csc_rowIdx, Z_csc_ColPtr, Z_csr_RowPtr;
	getCSC_CSR_from_COO(h_Z_values, h_Z_row_idx, h_Z_col_idx, &Z_csc_val,
			&Z_csc_rowIdx, &Z_csc_ColPtr, &Z_csr_val, &Z_csr_colIdx,
			&Z_csr_RowPtr, m, n);
	int nnz = h_Z_values.size();
	//==============MINIMIZATION PROCESS===============
	float meanvalue = 0;
	for (int i = 0; i < nnz; i++)
		meanvalue += h_Z_values[i];
	meanvalue = sqrt(meanvalue / nnz / r);
	//==============Create some meaningfull start point
	std::vector<float> h_L_object(r * m, meanvalue); // in ROW order
	std::vector<float> h_R_object(r * n, meanvalue); // in COL order
	float* h_L = &h_L_object[0];
	float* h_R = &h_R_object[0];
	//==============Residuals
	std::vector<int> residuals_csc_Pointers_object,
			residuals_csc_RowIdx_object, residuals_csc_ColPtr_object,
			residuals_csr_Pointers_object, residuals_csr_ColIdx_object,
			residuals_csr_RowPtr_object;
	std::vector<float> residuals_val_object(nnz, 0);
	float* residuals_val = &residuals_val_object[0];
	std::vector<int> residuals_idxes_object(nnz, 0);
	int* residuals_idxes = &residuals_idxes_object[0];
	for (int i = 0; i < nnz; i++) {
		residuals_idxes[i] = i;
		residuals_val[i] = -h_Z_values[i];
	}
	//get CSC and CSR from COO representation
	getCSC_CSR_from_COO(residuals_idxes_object, h_Z_row_idx, h_Z_col_idx,
			&residuals_csc_Pointers_object, &residuals_csc_RowIdx_object,
			&residuals_csc_ColPtr_object, &residuals_csr_Pointers_object,
			&residuals_csr_ColIdx_object, &residuals_csr_RowPtr_object, m, n);

	// Initialize data on device
	thrust::device_vector<float> d_L(h_L_object.begin(), h_L_object.end());// in ROW order
	thrust::device_vector<float> d_H(h_H_object.begin(), h_H_object.end());// in COL order

	thrust::device_vector<float> d_L_Lip_const_object(r * m, 0);// in ROW order
	thrust::device_vector<float> d_H_Lip_const_object(r * m, 0);// in COL order


//	residuals_csc_Pointers_object, &residuals_csc_RowIdx_object,
//				&residuals_csc_ColPtr_object, &residuals_csr_Pointers_object,
//				&residuals_csr_ColIdx_object, &residuals_csr_RowPtr_object




	//	    // initialize a device_vector with the list
	//	    thrust::device_vector<int> D(stl_list.begin(), stl_list.end());

	//	    // copy a device_vector into an STL vector
	//	    std::vector<int> stl_vector(D.size());
	//	    thrust::copy(D.begin(), D.end(), stl_vector.begin());

	std::vector<float> output_matrix(n * m, 0);
	*completed_matrix = output_matrix;
}
