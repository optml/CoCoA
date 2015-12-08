/*
 *  Matrix Completion Solver
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

void solveMCProblem(int m, int n, int r, std::vector<float> h_Z_values,
		std::vector<int> h_Z_col_idx, std::vector<int> h_Z_row_idx,
		std::vector<float>* completed_matrix, OptimizationSettings settings,
		float mu,float tv_penalty=0) {
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
	std::vector<int> residuals_csc_Pointers_object, residuals_csc_RowIdx_object,
			residuals_csc_ColPtr_object, residuals_csr_Pointers_object,
			residuals_csr_ColIdx_object, residuals_csr_RowPtr_object;
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
	//Get raw pointers to std::vector
	int* residuals_csc_Pointers = &residuals_csc_Pointers_object[0];
	int* residuals_csc_RowIdx = &residuals_csc_RowIdx_object[0];
	int* residuals_csc_ColPtr = &residuals_csc_ColPtr_object[0];
	int* residuals_csr_Pointers = &residuals_csr_Pointers_object[0];
	int* residuals_csr_ColIdx = &residuals_csr_ColIdx_object[0];
	int* residuals_csr_RowPtr = &residuals_csr_RowPtr_object[0];

	for (int i = 0; i < m; i++) {
		for (int id = residuals_csr_RowPtr[i]; id < residuals_csr_RowPtr[i + 1];
				id++) {
			int j = residuals_csr_ColIdx[id];
			for (int tmp = 0; tmp < r; tmp++) {
				residuals_val[residuals_csr_Pointers[id]] += h_L[i * r + tmp]
						* h_R[j * r + tmp];
			}
		}
	}
    // Compute actual Lipsch. constants
	std::vector<float> h_L_Lip_const_object(r * m, 0); // in ROW order
	std::vector<float> h_R_Lip_const_object(r * n, 0); // in COL order
	float* h_L_Lip_const = &h_L_Lip_const_object[0];
	float* h_R_Lip_const = &h_R_Lip_const_object[0];

	for (int u = 0; u < m; u++) {
		for (int i = 0; i < r; i++) {
			h_L_Lip_const[u * r + i] = 0;
			for (int j = Z_csr_RowPtr[u]; j < Z_csr_RowPtr[u + 1]; j++) {
				int v = Z_csr_colIdx[j];
				h_L_Lip_const[u * r + i] += 2 * h_R[v * r + i] * h_R[v * r + i];
			}
		}
	}

	for (int v = 0; v < n; v++) {
		for (int i = 0; i < r; i++) {
			h_R_Lip_const[v * r + i] = 0;
			for (int j = Z_csc_ColPtr[v]; j < Z_csc_ColPtr[v + 1]; j++) {
				int u = Z_csc_rowIdx[j];
				h_R_Lip_const[v * r + i] += 2 * h_L[u * r + i] * h_L[u * r + i];
			}
		}
	}

	float totalElapsedTime = 0;
	double startTimeWC;
	while (totalElapsedTime < settings.total_execution_time) {
		startTimeWC = gettime_();
		for (int it = 0;
				it < settings.iters_bulkIterations_count; it++) {
            //=======================Update Matrix L============================
			int u_cor = (int) (m * ((float) rand() / RAND_MAX));
			int i_cor = (int) (r * ((float) rand() / RAND_MAX));
            //compute partial derivative
			float partialDerivative = 0;
			for (int j = Z_csr_RowPtr[u_cor]; j < Z_csr_RowPtr[u_cor + 1];
					j++) {
				int v_cor = Z_csr_colIdx[j];
				partialDerivative += 2
						* residuals_val[residuals_csr_Pointers[j]]
						* h_R[v_cor * r + i_cor];
			}
			partialDerivative += mu * h_L[u_cor * r + i_cor];
			float delta = 0;
			if ((mu + h_L_Lip_const[u_cor * r + i_cor]) != 0) {
				delta = -partialDerivative
						/ (mu + h_L_Lip_const[u_cor * r + i_cor]);
				float chOfLipConst = h_L[u_cor * r + i_cor];
				h_L[u_cor * r + i_cor] += delta;
				for (int j = Z_csr_RowPtr[u_cor]; j < Z_csr_RowPtr[u_cor + 1];
						j++) {
					int v_cor = Z_csr_colIdx[j];
					residuals_val[residuals_csr_Pointers[j]] += delta
							* h_R[v_cor * r + i_cor];
				}
				chOfLipConst = h_L[u_cor * r + i_cor] * h_L[u_cor * r + i_cor]
						- chOfLipConst * chOfLipConst;
				chOfLipConst = 2 * chOfLipConst;
				for (int i = Z_csr_RowPtr[u_cor]; i < Z_csr_RowPtr[u_cor + 1];
						i++) {
					h_R_Lip_const[Z_csr_colIdx[i] * r + i_cor] += chOfLipConst;
				}

			}

            //===================== the same with matrix R======================
			int v_cor = (int) (n * ((float) rand() / RAND_MAX));
			i_cor = (int) (r * ((float) rand() / RAND_MAX));
			partialDerivative = 0;
			for (int j = Z_csc_ColPtr[v_cor]; j < Z_csc_ColPtr[v_cor + 1];
					j++) {
				u_cor = Z_csc_rowIdx[j];
				partialDerivative += 2
						* residuals_val[residuals_csc_Pointers[j]]
						* h_L[u_cor * r + i_cor];
			}
			partialDerivative += mu * h_R[v_cor * r + i_cor];
			delta = 0;
			if ((mu + h_R_Lip_const[v_cor * r + i_cor]) != 0) {
				delta = -partialDerivative
						/ (mu + h_R_Lip_const[v_cor * r + i_cor]);
				float chOfLipConst = h_R[v_cor * r + i_cor];
				h_R[v_cor * r + i_cor] += delta;
				chOfLipConst = h_R[v_cor * r + i_cor] * h_R[v_cor * r + i_cor]
						- chOfLipConst * chOfLipConst;
				chOfLipConst = 2 * chOfLipConst;
				for (int i = Z_csc_ColPtr[v_cor]; i < Z_csc_ColPtr[v_cor + 1];
						i++) {
					h_L_Lip_const[Z_csc_rowIdx[i] * r + i_cor] += chOfLipConst;
				}
				for (int j = Z_csc_ColPtr[v_cor]; j < Z_csc_ColPtr[v_cor + 1];
						j++) {
					u_cor = Z_csc_rowIdx[j];
					residuals_val[residuals_csc_Pointers[j]] += delta
							* h_L[u_cor * r + i_cor];
				}
			}
		}
		double endTimeWC = gettime_();
		totalElapsedTime += endTimeWC-startTimeWC;
		double totalResiduals = 0;
		for (int i = 0; i < nnz; i++)
			totalResiduals += residuals_val[i] * residuals_val[i];
		printf("Total %f  in %f sec\n", totalResiduals,
				totalElapsedTime);

	}
	double initialResiduals = 0;
	for (int i = 0; i < nnz; i++) {
		initialResiduals += h_Z_values[i] * h_Z_values[i];
	}
	printf("Initial residuals squared:%f \n", initialResiduals);
    //===========================Creating output============================
	std::vector<float> output_matrix(m * n, 0);
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			for (int i = 0; i < r; i++) {
				output_matrix[row * n + col] += h_L[row * r + i]
						* h_R[col * r + i];
			}
		}
	}
	*completed_matrix = output_matrix;
}
