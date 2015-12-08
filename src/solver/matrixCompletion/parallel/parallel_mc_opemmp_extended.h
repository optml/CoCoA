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
#include <omp.h>

template<typename L, typename D>
void run_matrix_completion_extended_openmp_solver(problem_mc_data<L, D> &ProblemData_inst,
		OptimizationSettings settings, OptimizationStatistics &statistics) {
	L m = ProblemData_inst.m;
	L n = ProblemData_inst.n;
	printf("check %d %d\n", ProblemData_inst.n, n);
	D mu = ProblemData_inst.mu;
	L rank = ProblemData_inst.rank;
	L nnz = ProblemData_inst.A_coo_values.size();
	//==============MINIMIZATION PROCESS===============
	D meanvalue = 0;
	for (int i = 0; i < nnz; i++)
		meanvalue += ProblemData_inst.A_coo_values[i];
	meanvalue = sqrt(meanvalue / nnz / rank);
	//==============Create some meaningfull start point
	if (settings.verbose)
		printf("size %d, %d,  %d %d %d\n", rank * m, rank * n, m, n, rank);
	ProblemData_inst.L_mat.resize(rank * m); // in ROW order
	ProblemData_inst.R_mat.resize(rank * n); // in COL order
	for (int i = 0; i < rank * m; i++) {
		ProblemData_inst.L_mat[i] = meanvalue * ((float) rand() / RAND_MAX);
	}
	for (int i = 0; i < rank * n; i++) {
		ProblemData_inst.R_mat[i] = meanvalue * ((float) rand() / RAND_MAX);
	}

	D* h_L = &ProblemData_inst.L_mat[0];
	D* h_R = &ProblemData_inst.R_mat[0];
	std::vector<L> residuals_csc_Pointers_object, residuals_csc_RowIdx_object, residuals_csc_ColPtr_object,
			residuals_csr_Pointers_object, residuals_csr_ColIdx_object, residuals_csr_RowPtr_object;
	std::vector<D> residuals_val_object(nnz, 0);
	D* residuals_val = &residuals_val_object[0];
	std::vector<L> residuals_idxes_object(nnz, 0);
	L* residuals_idxes = &residuals_idxes_object[0];
	//==============Residuals
	for (L i = 0; i < nnz; i++) {
		residuals_idxes[i] = i;
		residuals_val[i] = -ProblemData_inst.A_coo_values[i];
	}
	//get CSC and CSR from COO representation
	getCSC_CSR_from_COO(residuals_idxes_object, ProblemData_inst.A_coo_row_idx,
			ProblemData_inst.A_coo_col_idx, residuals_csc_Pointers_object, residuals_csc_RowIdx_object,
			residuals_csc_ColPtr_object, residuals_csr_Pointers_object, residuals_csr_ColIdx_object,
			residuals_csr_RowPtr_object, m, n);
	//Get raw pointers to std::vector
	L* residuals_csc_Pointers = &residuals_csc_Pointers_object[0];
	L* residuals_csc_RowIdx = &residuals_csc_RowIdx_object[0];
	L* residuals_csc_ColPtr = &residuals_csc_ColPtr_object[0];
	L* residuals_csr_Pointers = &residuals_csr_Pointers_object[0];
	L* residuals_csr_ColIdx = &residuals_csr_ColIdx_object[0];
	L* residuals_csr_RowPtr = &residuals_csr_RowPtr_object[0];

	for (L i = 0; i < m; i++) {
		for (L id = residuals_csr_RowPtr[i]; id < residuals_csr_RowPtr[i + 1]; id++) {
			L j = residuals_csr_ColIdx[id];
			for (L tmp = 0; tmp < rank; tmp++) {
				residuals_val[residuals_csr_Pointers[id]] += h_L[i * rank + tmp] * h_R[j * rank + tmp];
			}
		}
	}
	// Compute actual Lipsch. constants
	std::vector<D> h_L_Lip_const_object(rank * m, 0); // in ROW order
	std::vector<D> h_R_Lip_const_object(rank * n, 0); // in COL order
	D* h_L_Lip_const = &h_L_Lip_const_object[0];
	D* h_R_Lip_const = &h_R_Lip_const_object[0];

	for (L u = 0; u < m; u++) {
		for (L i = 0; i < rank; i++) {
			h_L_Lip_const[u * rank + i] = 0;
			for (L j = residuals_csr_RowPtr[u]; j < residuals_csr_RowPtr[u + 1]; j++) {
				L v = residuals_csr_ColIdx[j];
				h_L_Lip_const[u * rank + i] += 2 * h_R[v * rank + i] * h_R[v * rank + i];
			}
		}
	}
	for (L v = 0; v < n; v++) {
		for (L i = 0; i < rank; i++) {
			h_R_Lip_const[v * rank + i] = 0;
			for (L j = residuals_csc_ColPtr[v]; j < residuals_csc_ColPtr[v + 1]; j++) {
				L u = residuals_csc_RowIdx[j];
				h_R_Lip_const[v * rank + i] += 2 * h_L[u * rank + i] * h_L[u * rank + i];
			}
		}
	}

	D initialResiduals = 0;
	if (settings.verbose) {
		for (L i = 0; i < nnz; i++) {
			initialResiduals += ProblemData_inst.A_coo_values[i] * ProblemData_inst.A_coo_values[i];
		}
		printf("Initial residuals squared:%f \n", initialResiduals);
	}

	//============Initialization random states
	int totalThreds = 0;
	omp_set_num_threads(settings.totalThreads);
	L total_full_iterations = 0;
#pragma omp parallel  shared(totalThreds)
	{
		totalThreds = omp_get_num_threads();
	}
	if (settings.verbose)
		printf("Total threads %d \n", totalThreds);
	unsigned int s;
	unsigned int seed[totalThreds];
	for (int i = 0; i < totalThreds; i++) {
		seed[i] = (int) RAND_MAX * rand();
		if (seed[i] < 0)
			seed[i] = -seed[i];
	}

	double totalElapsedTime = 0;

	while (totalElapsedTime < settings.total_execution_time) {

		double start_time = gettime_();
		//		/*
		for (int it = 0; it < settings.iters_bulkIterations_count; it++) {
			//=======================Update Matrix L============================
#pragma omp parallel
			{
				s = seed[omp_get_thread_num()];
#pragma omp for schedule(dynamic,2)
				for (L u_cor = 0; u_cor < m; u_cor++) {
					L i_cor = (L) (rank * ((float) rand_r(&s) / RAND_MAX));
					//compute partial derivative
					D partialDerivative = 0;
					for (L j = residuals_csr_RowPtr[u_cor]; j < residuals_csr_RowPtr[u_cor + 1]; j++) {
						L v_cor = residuals_csr_ColIdx[j];
						L id_to_data = residuals_csr_Pointers[j];
						if (residuals_val[id_to_data] * ProblemData_inst.A_coo_operator[id_to_data] >= 0) {
							partialDerivative += 2 * residuals_val[id_to_data] * h_R[v_cor * rank + i_cor];
						}
					}
					partialDerivative += mu * h_L[u_cor * rank + i_cor];
					D delta = 0;
					if ((mu + h_L_Lip_const[u_cor * rank + i_cor]) != 0) {
						delta = -partialDerivative / (mu + h_L_Lip_const[u_cor * rank + i_cor]);
						D chOfLipConst = h_L[u_cor * rank + i_cor];
						h_L[u_cor * rank + i_cor] += delta;
						for (L j = residuals_csr_RowPtr[u_cor]; j < residuals_csr_RowPtr[u_cor + 1]; j++) {
							L v_cor = residuals_csr_ColIdx[j];
							residuals_val[residuals_csr_Pointers[j]] += delta * h_R[v_cor * rank + i_cor];
						}
						chOfLipConst = h_L[u_cor * rank + i_cor] * h_L[u_cor * rank + i_cor] - chOfLipConst
								* chOfLipConst;
						chOfLipConst = 2 * chOfLipConst;
						for (L i = residuals_csr_RowPtr[u_cor]; i < residuals_csr_RowPtr[u_cor + 1]; i++) {
							h_R_Lip_const[residuals_csr_ColIdx[i] * rank + i_cor] += chOfLipConst;
						}
					}
				}
				//===================== the same with matrix R======================
#pragma omp for  schedule(dynamic,2)
				for (L v_cor = 0; v_cor < n; v_cor++) {
					L i_cor = (L) (rank * ((float) rand_r(&s) / RAND_MAX));
					D partialDerivative = 0;
					for (L j = residuals_csc_ColPtr[v_cor]; j < residuals_csc_ColPtr[v_cor + 1]; j++) {
						L u_cor = residuals_csc_RowIdx[j];
						L id_to_data = residuals_csc_Pointers[j];
						if (residuals_val[id_to_data] * ProblemData_inst.A_coo_operator[id_to_data] >= 0) {
							partialDerivative += 2 * residuals_val[id_to_data] * h_L[u_cor * rank + i_cor];
						}
					}
					partialDerivative += mu * h_R[v_cor * rank + i_cor];
					D delta = 0;
					if ((mu + h_R_Lip_const[v_cor * rank + i_cor]) != 0) {
						delta = -partialDerivative / (mu + h_R_Lip_const[v_cor * rank + i_cor]);
						D chOfLipConst = h_R[v_cor * rank + i_cor];
						h_R[v_cor * rank + i_cor] += delta;
						chOfLipConst = h_R[v_cor * rank + i_cor] * h_R[v_cor * rank + i_cor] - chOfLipConst
								* chOfLipConst;
						chOfLipConst = 2 * chOfLipConst;
						for (L i = residuals_csc_ColPtr[v_cor]; i < residuals_csc_ColPtr[v_cor + 1]; i++) {
							h_L_Lip_const[residuals_csc_RowIdx[i] * rank + i_cor] += chOfLipConst;
						}
						for (L j = residuals_csc_ColPtr[v_cor]; j < residuals_csc_ColPtr[v_cor + 1]; j++) {
							L u_cor = residuals_csc_RowIdx[j];
							residuals_val[residuals_csc_Pointers[j]] += delta * h_L[u_cor * rank + i_cor];
						}
					}
				}

				seed[omp_get_thread_num()] = s;
			}
		}
		double end_time = gettime_();
		total_full_iterations++;
		totalElapsedTime += end_time - start_time;
		if (settings.verbose) {
			printf("Last loop took %f sec", end_time - start_time);
			D totalResiduals = 0;
			for (L i = 0; i < nnz; i++) {
				if (residuals_val[i] * ProblemData_inst.A_coo_operator[i] >= 0) {
					totalResiduals += residuals_val[i] * residuals_val[i];
				}
			}
			printf("Total residuals %f  elapsed time %f sec\n", totalResiduals, totalElapsedTime);
		}
	}
	if (settings.verbose) {
		printf("Initial residuals squared:%f \n", initialResiduals);
	}

	statistics.number_of_iters_in_millions = (double) total_full_iterations * (n + m) / 1000000;
	statistics.average_speed_iters_per_ms = total_full_iterations * (n + m) / (totalElapsedTime * 1000);
	statistics.elapsed_time = totalElapsedTime * 1000;
	//===========================Creating output============================
	//	std::vector<float> output_matrix(m * n, 0);
	//	for (int row = 0; row < m; row++) {
	//		for (int col = 0; col < n; col++) {
	//			for (int i = 0; i < r; i++) {
	//				output_matrix[row * n + col] += h_L[row * r + i] * h_R[col * r
	//						+ i];
	//			}
	//		}
	//	}
}
