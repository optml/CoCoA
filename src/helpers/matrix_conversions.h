/* The following functions are for converting between different Matrix sparse representations
 *
 * COO - Coordinate representation
 *       Value, Row_IDX, Col_IDX
 * CSC - Column wise sparse representation
 *       Values and Row_IDX is in column order,
 *       A_csc_ColPtr  is pointer to array when given column starts
 * CSR - Simillar as CSC but row-wise
 *
 */

#ifndef MATRIX_CONVERSIONS_H_
#define MATRIX_CONVERSIONS_H_

template<typename L, typename D>
void getCSC_CSR_from_COO(std::vector<D> h_Z_values, std::vector<L> h_Z_row_idx, std::vector<L> h_Z_col_idx,
		std::vector<D>& Z_csc_val, std::vector<L>& Z_csc_rowIdx, std::vector<L>& Z_csc_ColPtr,
		std::vector<D>& Z_csr_val, std::vector<L>& Z_csr_colIdx, std::vector<L>& Z_csr_RowPtr, L m, L n) {
	L nnz = h_Z_values.size();
	Z_csc_val.resize(nnz, 0);
	Z_csc_rowIdx.resize(nnz, 0);
	Z_csc_ColPtr.resize(n + 1, 0);
	Z_csr_val.resize(nnz, 0);
	Z_csr_colIdx.resize(nnz, 0);
	Z_csr_RowPtr.resize(m + 1, 0);
	for (L i = 0; i < nnz; i++) {
		Z_csr_RowPtr[h_Z_row_idx[i]]++;
		Z_csc_ColPtr[h_Z_col_idx[i]]++;
	}
	// Get Z matrix in CSC and get Z matrix in RSC
	for (L i = 0; i < m; i++) {
		Z_csr_RowPtr[i + 1] += Z_csr_RowPtr[i];
	}
	for (L i = m; i > 0; i--) {
		Z_csr_RowPtr[i] = Z_csr_RowPtr[i - 1];
	}
	Z_csr_RowPtr[0] = 0;
	//	// Get the same for csc
	for (L i = 0; i < n; i++) {
		Z_csc_ColPtr[i + 1] += Z_csc_ColPtr[i];
	}
	for (L i = n; i > 0; i--) {
		Z_csc_ColPtr[i] = Z_csc_ColPtr[i - 1];
	}
	Z_csc_ColPtr[0] = 0;
	// ========Fill Data into correct format
	for (L i = 0; i < nnz; i++) {
		Z_csc_val[Z_csc_ColPtr[h_Z_col_idx[i]]] = h_Z_values[i];
		Z_csc_rowIdx[Z_csc_ColPtr[h_Z_col_idx[i]]] = h_Z_row_idx[i];
		Z_csc_ColPtr[h_Z_col_idx[i]]++;
		Z_csr_val[Z_csr_RowPtr[h_Z_row_idx[i]]] = h_Z_values[i];
		Z_csr_colIdx[Z_csr_RowPtr[h_Z_row_idx[i]]] = h_Z_col_idx[i];
		Z_csr_RowPtr[h_Z_row_idx[i]]++;
	}
	for (L i = m; i > 0; i--) {
		Z_csr_RowPtr[i] = Z_csr_RowPtr[i - 1];
	}
	Z_csr_RowPtr[0] = 0;
	for (L i = n; i > 0; i--) {
		Z_csc_ColPtr[i] = Z_csc_ColPtr[i - 1];
	}
	Z_csc_ColPtr[0] = 0;
}

template<typename T, typename I>
void getCSR_from_CSC(const std::vector<T>& Z_csc_values, //Input
		const std::vector<I> &Z_csc_row_idx, const std::vector<I>& Z_csc_col_ptr, std::vector<T>& Z_csr_val,//Output
		std::vector<I>& Z_csr_colIdx, std::vector<I>& Z_csr_RowPtr, I m, I n) {
	I nnz = Z_csc_values.size();
	Z_csr_val.resize(nnz, 0);
	Z_csr_colIdx.resize(nnz, 0);
	Z_csr_RowPtr.resize(m + 1, 0);
	for (int i=0;i<Z_csr_RowPtr.size();i++)
		Z_csr_RowPtr[i]=0;
	for (I i = 0; i < nnz; i++) {
		Z_csr_RowPtr[Z_csc_row_idx[i]]++;
	}


	//   get Z matrix in RSC
	for (I i = 0; i < m; i++) {
		Z_csr_RowPtr[i + 1] += Z_csr_RowPtr[i];
	}
	for (I i = m; i > 0; i--) {
		Z_csr_RowPtr[i] = Z_csr_RowPtr[i - 1];
	}
	Z_csr_RowPtr[0] = 0;
	// ========Fill Data into correct format


	for (I col = 0; col < n; col++) {
		for (I tmp = Z_csc_col_ptr[col]; tmp < Z_csc_col_ptr[col + 1]; tmp++) {
			I row_id = Z_csc_row_idx[tmp];
			Z_csr_val[Z_csr_RowPtr[row_id]] = Z_csc_values[tmp];
			Z_csr_colIdx[Z_csr_RowPtr[row_id]] = col;
			Z_csr_RowPtr[row_id]++;
		}
	}

	for (I i = m; i > 0; i--) {
		Z_csr_RowPtr[i] = Z_csr_RowPtr[i - 1];
	}
	Z_csr_RowPtr[0] = 0;
}

void getCSR_from_CSC(const std::vector<int> &Z_csc_row_idx, const std::vector<int>& Z_csc_col_ptr,
		std::vector<int>& Z_csr_colIdx, std::vector<int>& Z_csr_RowPtr, int m, int n, int nnz) {
	Z_csr_colIdx.resize(nnz, 0);
	Z_csr_RowPtr.resize(m + 1, 0);
	for (int i = 0; i < nnz; i++) {
		Z_csr_RowPtr[Z_csc_row_idx[i]]++;
	}
	//   get Z matrix in RSC
	for (int i = 0; i < m; i++) {
		Z_csr_RowPtr[i + 1] += Z_csr_RowPtr[i];
	}
	for (int i = m; i > 0; i--) {
		Z_csr_RowPtr[i] = Z_csr_RowPtr[i - 1];
	}
	Z_csr_RowPtr[0] = 0;
	// ========Fill Data into correct format
	for (int col = 0; col < n; col++) {
		for (int tmp = Z_csc_col_ptr[col]; tmp < Z_csc_col_ptr[col + 1]; tmp++) {
			int row_id = Z_csc_row_idx[tmp];
			Z_csr_colIdx[Z_csr_RowPtr[row_id]] = col;
			Z_csr_RowPtr[row_id]++;
		}
	}

	for (int i = m; i > 0; i--) {
		Z_csr_RowPtr[i] = Z_csr_RowPtr[i - 1];
	}
	Z_csr_RowPtr[0] = 0;
}

template<typename T, typename I>
void getCSR_from_COO(std::vector<T> &h_Z_values, std::vector<I> &h_Z_row_idx, std::vector<I> &h_Z_col_idx,
		std::vector<T> &Z_csr_val, std::vector<I> &Z_csr_colIdx, std::vector<I>& Z_csr_RowPtr, int m, int n) {
	long long nnz = h_Z_values.size();

//	cout << "Z "<< nonzero_elements_of_input_data<< endl;
	Z_csr_val.resize(nnz, 0);
	Z_csr_colIdx.resize(nnz, 0);
	Z_csr_RowPtr.resize(m + 1, 0);

	for (long long i = 0; i < nnz; i++) {
		Z_csr_RowPtr[h_Z_row_idx[i]]++;
	}
	// Get Z matrix in CSC and get Z matrix in RSC
	for (long long i = 0; i < m; i++) {
		Z_csr_RowPtr[i + 1] += Z_csr_RowPtr[i];
	}
	for (long long i = m; i > 0; i--) {
		Z_csr_RowPtr[i] = Z_csr_RowPtr[i - 1];
	}
	Z_csr_RowPtr[0] = 0;
	// ========Fill Data into correct format
	for (long long i = 0; i < nnz; i++) {
		Z_csr_val[Z_csr_RowPtr[h_Z_row_idx[i]]] = h_Z_values[i];
		Z_csr_colIdx[Z_csr_RowPtr[h_Z_row_idx[i]]] = h_Z_col_idx[i];
		Z_csr_RowPtr[h_Z_row_idx[i]]++;
	}
	for (int i = m; i > 0; i--) {
		Z_csr_RowPtr[i] = Z_csr_RowPtr[i - 1];
	}
	Z_csr_RowPtr[0] = 0;
}

//template<typename T, typename I>
//void get_COO_from_CSR(conststd::vector<T>& Z_csr_val, const std::vector<I>& Z_csr_col_idx, const std::vector<
//		I>& Z_csr_row_ptr, std::vector<T> &A_coo_val_idx, std::vector<I> &A_coo_row_idx,
//		std::vector<I>& A_coo_col_idx) {
//	A_coo_col_idx.resize(Z_csr_col_idx.size());
//	A_coo_val_idx.resize(Z_csr_col_idx.size());
//	A_coo_row_idx.resize(Z_csr_col_idx.size());
//	I nnz = 0;
//	for (I row = 0; row < Z_csr_row_ptr.size() - 1; row++) {
//		for (I col_tmp = Z_csr_row_ptr[row]; col_tmp < Z_csr_row_ptr[row + 1]; col_tmp++) {
//			A_coo_col_idx[nnz] = Z_csr_col_idx[col_tmp];
//			A_coo_row_idx[nnz] = row;
//			A_coo_val[nnz] = Z_csr_val[col_tmp];
//			nnz++;
//		}
//	}
//}

template<typename T, typename I>
void getCSC_from_COO(const std::vector<T> &h_Z_values, const std::vector<I> &h_Z_row_idx,
		const std::vector<I>& h_Z_col_idx, std::vector<T>& Z_csc_val, std::vector<I>& Z_csc_rowIdx,
		std::vector<I>& Z_csc_ColPtr, int m, int n) {
	unsigned long long nnz = h_Z_values.size();
	Z_csc_val.resize(nnz);
	Z_csc_rowIdx.resize(nnz);
	Z_csc_ColPtr.resize(n + 1);
	for (int i = 0; i < nnz; i++) {
		Z_csc_ColPtr[h_Z_col_idx[i]]++;
	}
	//	// Get the same for csc
	for (int i = 0; i < n; i++) {
		Z_csc_ColPtr[i + 1] += Z_csc_ColPtr[i];
	}
	for (int i = n; i > 0; i--) {
		Z_csc_ColPtr[i] = Z_csc_ColPtr[i - 1];
	}
	Z_csc_ColPtr[0] = 0;
	// ========Fill Data into correct format
	for (int i = 0; i < nnz; i++) {
		Z_csc_val[Z_csc_ColPtr[h_Z_col_idx[i]]] = h_Z_values[i];
		Z_csc_rowIdx[Z_csc_ColPtr[h_Z_col_idx[i]]] = h_Z_row_idx[i];
		Z_csc_ColPtr[h_Z_col_idx[i]]++;
	}
	for (int i = n; i > 0; i--) {
		Z_csc_ColPtr[i] = Z_csc_ColPtr[i - 1];
	}
	Z_csc_ColPtr[0] = 0;
}

#endif /* MATRIX_CONVERSIONS_H_ */
