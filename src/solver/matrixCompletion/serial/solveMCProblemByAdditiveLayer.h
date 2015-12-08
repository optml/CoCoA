/*
 *  Matrix Completion Solver via Additive Layer
 *
 * m - number of rows
 * n - number of cols
 * maximal_rank - maximal rank of matrix which we want to find
 * h_Z_values - known values
 * h_Z_col_idx - column index of known data
 * h_Z_row_idx - row index of known data
 * (h_Z_.....) is a sparse matrix in COO format
 * completed_matrix - is reconstructed matrix saved in
 *                    row oriented format
 * settings - setting to optimizer
 * outFileName - file name (without extension) where all reconstructions
 *               until final rank are saved.
 * toll - tollerance when to stop or increate rank
 */

using namespace std;
#include <vector>
#include "../../structures.h"
#include "../../../helpers/matrix_conversions.h"
#include "../../../helpers/time_functions.h"
#include "../../../helpers/csv_writter.h"
#include "../../../mc/mc_problem_generation.h"

void solveMCProblemByAdditiveLayer(int m, int n, int maximal_rank,
		std::vector<float> h_Z_values, std::vector<int> h_Z_col_idx,
		std::vector<int> h_Z_row_idx, std::vector<float>* completed_matrix,
		OptimizationSettings settings, float mu, const char* outFileName,
		float toll = 0.0001) {
	std::vector<float> Z_csr_val, Z_csc_val;
	std::vector<int> Z_csr_colIdx, Z_csc_rowIdx, Z_csc_ColPtr, Z_csr_RowPtr;
	getCSC_CSR_from_COO(h_Z_values, h_Z_row_idx, h_Z_col_idx, &Z_csc_val,
			&Z_csc_rowIdx, &Z_csc_ColPtr, &Z_csr_val, &Z_csr_colIdx,
			&Z_csr_RowPtr, m, n);
	int nnz = h_Z_values.size();
	std::vector<float> output_matrix(m * n, 0);

	std::vector<int> residuals_csc_Pointers, residuals_csc_RowIdx,
			residuals_csc_ColPtr, residuals_csr_Pointers, residuals_csr_ColIdx,
			residuals_csr_RowPtr;
	std::vector<float> residuals_val(nnz, 0);
	float lastRezidualVal = 0;
	std::vector<int> residuals_idxes(nnz, 0);
	for (int i = 0; i < nnz; i++) {
		residuals_idxes[i] = i;
		residuals_val[i] = -h_Z_values[i];
		lastRezidualVal += residuals_val[i] * residuals_val[i];
	}
	getCSC_CSR_from_COO(residuals_idxes, h_Z_row_idx, h_Z_col_idx,
			&residuals_csc_Pointers, &residuals_csc_RowIdx,
			&residuals_csc_ColPtr, &residuals_csr_Pointers,
			&residuals_csr_ColIdx, &residuals_csr_RowPtr, m, n);

	int r = 0;

	clock_t startTime = clock();
	while (lastRezidualVal > toll && r < maximal_rank) {
		printf("Current rank:%d \n", r);
		float meanvalue = 0;
		for (int i = 0; i < nnz; i++) {
			meanvalue += h_Z_values[i]
					- output_matrix[h_Z_row_idx[i] * n + h_Z_col_idx[i]];
		}
		meanvalue = sqrt(abs(meanvalue) / nnz);

		//==============Create some meaningfull start point
		std::vector<float> h_L(m, meanvalue); // in ROW order
		std::vector<float> h_R(n, meanvalue); // in COL order

		for (int i = 0; i < nnz; i++) {
			residuals_val[i] = -h_Z_values[i]
					+ output_matrix[h_Z_row_idx[i] * n + h_Z_col_idx[i]];
		}

		for (int i = 0; i < m; i++) {
			for (int id = residuals_csr_RowPtr[i];
					id < residuals_csr_RowPtr[i + 1]; id++) {
				int j = residuals_csr_ColIdx[id];
				for (int tmp = 0; tmp < 1; tmp++) {
					residuals_val[residuals_csr_Pointers[id]] += h_L[i + tmp]
							* h_R[j + tmp];
				}
			}
		}

		// Compute actial Lipsch. constants
		std::vector<float> h_L_Lip_const(m, 0); // in ROW order
		std::vector<float> h_R_Lip_const(n, 0); // in COL order

		for (int u = 0; u < m; u++) {
			h_L_Lip_const[u] = 0;
			for (int j = Z_csr_RowPtr[u]; j < Z_csr_RowPtr[u + 1]; j++) {
				int v = Z_csr_colIdx[j];
				h_L_Lip_const[u] += 2 * h_R[v] * h_R[v];
			}
		}

		for (int v = 0; v < n; v++) {
			h_R_Lip_const[v] = 0;
			for (int j = Z_csc_ColPtr[v]; j < Z_csc_ColPtr[v + 1]; j++) {
				int u = Z_csc_rowIdx[j];
				h_R_Lip_const[v] += 2 * h_L[u] * h_L[u];
			}
		}

		double totalResiduals = 0;

		while (abs(lastRezidualVal - totalResiduals) > toll) {
			lastRezidualVal = totalResiduals;
			for (int it = 0;
					it < settings.iters_bulkIterations_count;
					it++) {
//Update Matrix L
				int u_cor = (int) (m * ((float) rand() / RAND_MAX));
//compute partial derivative
				float partialDerivative = 0;
				for (int j = Z_csr_RowPtr[u_cor]; j < Z_csr_RowPtr[u_cor + 1];
						j++) {
					int v_cor = Z_csr_colIdx[j];
					partialDerivative += 2
							* residuals_val[residuals_csr_Pointers[j]]
							* h_R[v_cor];
				}
				partialDerivative += mu * h_L[u_cor];
				float delta = 0;
				if ((mu + h_L_Lip_const[u_cor]) != 0) {
					delta = -partialDerivative / (mu + h_L_Lip_const[u_cor]);
					float chOfLipConst = h_L[u_cor];
					h_L[u_cor] += delta;
					for (int j = Z_csr_RowPtr[u_cor];
							j < Z_csr_RowPtr[u_cor + 1]; j++) {
						int v_cor = Z_csr_colIdx[j];
						residuals_val[residuals_csr_Pointers[j]] += delta
								* h_R[v_cor];
					}
					chOfLipConst = h_L[u_cor] * h_L[u_cor]
							- chOfLipConst * chOfLipConst;
					chOfLipConst = 2 * chOfLipConst;
					for (int i = Z_csr_RowPtr[u_cor];
							i < Z_csr_RowPtr[u_cor + 1]; i++) {
						h_R_Lip_const[Z_csr_colIdx[i]] += chOfLipConst;
					}

				}

//===================================================
//		// the same with matrix R
				int v_cor = (int) (n * ((float) rand() / RAND_MAX));
				partialDerivative = 0;
				for (int j = Z_csc_ColPtr[v_cor]; j < Z_csc_ColPtr[v_cor + 1];
						j++) {
					u_cor = Z_csc_rowIdx[j];
					partialDerivative += 2
							* residuals_val[residuals_csc_Pointers[j]]
							* h_L[u_cor];
				}
				partialDerivative += mu * h_R[v_cor];
				delta = 0;
				if ((mu + h_R_Lip_const[v_cor]) != 0) {
					delta = -partialDerivative / (mu + h_R_Lip_const[v_cor]);
					float chOfLipConst = h_R[v_cor];
					h_R[v_cor] += delta;
					chOfLipConst = h_R[v_cor] * h_R[v_cor]
							- chOfLipConst * chOfLipConst;
					chOfLipConst = 2 * chOfLipConst;
					for (int i = Z_csc_ColPtr[v_cor];
							i < Z_csc_ColPtr[v_cor + 1]; i++) {
						h_L_Lip_const[Z_csc_rowIdx[i]] += chOfLipConst;
					}
					for (int j = Z_csc_ColPtr[v_cor];
							j < Z_csc_ColPtr[v_cor + 1]; j++) {
						u_cor = Z_csc_rowIdx[j];
						residuals_val[residuals_csc_Pointers[j]] += delta
								* h_L[u_cor];
					}
				}
			}
			totalResiduals = 0;
			for (int i = 0; i < nnz; i++)
				totalResiduals += residuals_val[i] * residuals_val[i];
			printf("Total %f  in %f sec\n", totalResiduals,
					getTotalElapsetTime(&startTime));
		}

		for (int row = 0; row < m; row++) {
			for (int col = 0; col < n; col++) {
				output_matrix[row * n + col] += h_L[row] * h_R[col];
			}
		}
		char outputFileName[500];
		sprintf(outputFileName, "%s_%d.csv", outFileName, r);
		saveDataToCSVFile(m, n, outputFileName, output_matrix);
		r++;
	}

	double initialResiduals = 0;
	for (int i = 0; i < nnz; i++) {
		initialResiduals += h_Z_values[i] * h_Z_values[i];
	}
	printf("Initial residuals squared:%f \n", initialResiduals);

	*completed_matrix = output_matrix;
}
