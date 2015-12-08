#include <vector>
#include <iostream>
//#include <zoltan_cpp.h>

#include "../../helpers/matrix_conversions.h"

// An adaptor from CSC to Zoltan's hypergraph interface
template<typename L, typename D>
class hypergraph {

public:
	L cols;
	L rows;
	L nnz;
	L my_rank;
	// ZOLTAN_COMPRESSED_VERTEX / Compressed Sparse Column
	const std::vector<L> &row_idx;
	const std::vector<L> &col_ptr;

	//public:
	// a useless constructor
	hypergraph() {
		cols = 0;
		rows = 0;
		nnz = 0;
	}
	// the real constructor
	hypergraph(const std::vector<L> &A_csc_row_idx, const std::vector<L> &A_csc_col_ptr, const L m, const L n,
			const L _my_rank) :
		cols(n), rows(m), row_idx(A_csc_row_idx), col_ptr(A_csc_col_ptr), my_rank(_my_rank) {
		nnz = A_csc_row_idx.size();
	}
	// destructor
	~hypergraph() {
	}

	static int get_number_of_vertices(void *data, int *ierr) {
		*ierr = ZOLTAN_OK;
		hypergraph *hg = (hypergraph *) data;
		return hg->cols;
	}

	static void get_vertex_list(void *data, int sizeGID, int sizeLID, ZOLTAN_ID_PTR globalID,
			ZOLTAN_ID_PTR localID, int wgt_dim, float *obj_wgts, int *ierr) {
		*ierr = ZOLTAN_OK;
		hypergraph *hg = (hypergraph *) data;
		for (int i = 0; i < hg->cols; i++) {
			globalID[i] = i;
			localID[i] = i;
		}
	}

	static void get_hypergraph_size(void *data, int *num_lists, int *num_nonzeroes, int *format, int *ierr) {
		*ierr = ZOLTAN_OK;
		hypergraph *hg = (hypergraph *) data;
		*num_lists = hg->rows;
		*num_nonzeroes = hg->nnz;
		*format = ZOLTAN_COMPRESSED_EDGE;
	}

	static void get_hypergraph(void *data, int sizeGID, int num_vertices, int num_nonzeroes, int format,
			ZOLTAN_ID_PTR edgeGID, int *vtxPtr, ZOLTAN_ID_PTR vtxGID, int *ierr) {

		*ierr = ZOLTAN_OK;
		hypergraph *hg = (hypergraph *) data;

		if ((num_vertices != hg->rows) || (num_nonzeroes != hg->nnz) || (format != ZOLTAN_COMPRESSED_EDGE)) {
			*ierr = ZOLTAN_FATAL;
			return;
		}

		std::vector<L> Z_csr_colIdx;
		std::vector<L> Z_csr_RowPtr;

		//		printf("Number od Edges %d  Number of verticesIDX %d\n", hg->rows, hg->nnz);

		getCSR_from_CSC(hg->row_idx, hg->col_ptr, Z_csr_colIdx, Z_csr_RowPtr, hg->rows, hg->cols, hg->nnz);

		//		//Print CSC data
		//		for (L i = 0; i < hg->col_ptr.size() - 1; i++) {
		//			printf("Vertex %d: ", i);
		//			for (L j = hg->col_ptr[i]; j < hg->col_ptr[i + 1]; j++) {
		//				printf("%d ", hg->row_idx[j]);
		//			}
		//			printf("\n");
		//		}
		//
		//				Print CSR data
		//		for (L i = 0; i < Z_csr_RowPtr.size() - 1; i++) {
		//			printf("RANK%d EDGE %d: ", hg->my_rank, i);
		//			for (L j = Z_csr_RowPtr[i]; j < Z_csr_RowPtr[i + 1]; j++) {
		//				printf("%d ", Z_csr_colIdx[j]);
		//			}
		//			printf("\n");
		//		}

		// Checking
		//		for (L i = 0; i < hg->nnz; i++) {
		//			if (Z_csr_colIdx[i] > hg->cols)
		//				printf("========== ZLE DATA=======\n"); // was: hg->row_idx
		//		}

		// TODO: This needs to be rewritten -- we have the data already!
		int i;
		for (i = 0; i < hg->rows; i++) {
			edgeGID[i] = i;
			vtxPtr[i] = Z_csr_RowPtr[i]; // was: hg->col_ptr
		}
		for (i = 0; i < hg->nnz; i++) {
			vtxGID[i] = Z_csr_colIdx[i]; // was: hg->row_idx
		}

		//		std::cout << "FINISHED OK" << std::endl;
	}

};

template<typename L, typename D>
class hypergraph_in_CSR {

public:
	L cols;
	L rows;
	L nnz;
	L my_rank;
	L total_number_of_vertices;
	// ZOLTAN_COMPRESSED_VERTEX / Compressed Sparse Column
	const std::vector<L> &row_ptr;
	const std::vector<L> &col_idx;
	std::vector<L> m_parts;
	//public:
	L my_vertices;

	const std::vector<L> &columns_parts;

	// a useless constructor
	hypergraph_in_CSR() {
		cols = 0;
		rows = 0;
		nnz = 0;
	}
	// the real constructor
	hypergraph_in_CSR(const std::vector<L> &A_csr_col_idx, const std::vector<L> &A_csr_row_ptr, const L m,
			const L n, const L _my_rank, const L _total_number_of_vertices, const std::vector<L> _m_parts,
			const std::vector<L> &_columns_parts) :
		cols(n), rows(m), col_idx(A_csr_col_idx), row_ptr(A_csr_row_ptr), my_rank(_my_rank),
				total_number_of_vertices(_total_number_of_vertices), columns_parts (_columns_parts) {
		nnz = A_csr_col_idx.size();
		m_parts = _m_parts;

		my_vertices = 0;
		for (L i = 0; i < columns_parts.size(); i++) {
			if (columns_parts[i] == my_rank) {
				my_vertices++;
			}
		}

	}
	// destructor
	~hypergraph_in_CSR() {
	}

	static int get_number_of_vertices(void *data, int *ierr) {
		*ierr = ZOLTAN_OK;
		hypergraph_in_CSR *hg = (hypergraph_in_CSR *) data;

		return hg->my_vertices;

		if (hg->my_rank == 0) {
			return hg->total_number_of_vertices;
		} else
			return 0;
	}

	static void get_vertex_list(void *data, int sizeGID, int sizeLID, ZOLTAN_ID_PTR globalID,
			ZOLTAN_ID_PTR localID, int wgt_dim, float *obj_wgts, int *ierr) {
		*ierr = ZOLTAN_OK;
		hypergraph_in_CSR *hg = (hypergraph_in_CSR *) data;

		L j = 0;
		for (int i = 0; i < hg->columns_parts.size(); i++) {
			if (hg->columns_parts[i] == hg->my_rank) {
				globalID[j] = i;
				localID[j] = i;
				j++;
			}
		}

		//		if (hg->my_rank == 0) {
		//			for (int i = 0; i < hg->total_number_of_vertices; i++) {
		//				globalID[i] = i;
		//				localID[i] = i;
		//			}
		//		}
	}

	static void get_hypergraph_size(void *data, int *num_lists, int *num_nonzeroes, int *format, int *ierr) {
		*ierr = ZOLTAN_OK;
		hypergraph_in_CSR *hg = (hypergraph_in_CSR *) data;
		*num_lists = hg->rows;
		*num_nonzeroes = hg->nnz;
		//				printf("Ja som node %d a mam lis%d a nnz %d  total N%d\n", hg->my_rank, hg->rows, hg->nnz,hg->total_number_of_vertices);
		*format = ZOLTAN_COMPRESSED_EDGE;
	}

	static void get_hypergraph(void *data, int sizeGID, int num_vertices, int num_nonzeroes, int format,
			ZOLTAN_ID_PTR edgeGID, int *vtxPtr, ZOLTAN_ID_PTR vtxGID, int *ierr) {

		*ierr = ZOLTAN_OK;
		hypergraph_in_CSR *hg = (hypergraph_in_CSR *) data;

		if ((num_vertices != hg->rows) || (num_nonzeroes != hg->nnz) || (format != ZOLTAN_COMPRESSED_EDGE)) {
			*ierr = ZOLTAN_FATAL;
			return;
		}
		std::vector<L> Z_csr_colIdx;
		std::vector<L> Z_csr_RowPtr;
		int i;
		//Check my local hypergraph
		for (i = 0; i < hg->rows; i++) {
			for (int tmp = hg->row_ptr[i]; tmp < hg->row_ptr[i + 1]; tmp++) {
				if ((hg->col_idx[tmp] >= hg->total_number_of_vertices) || (hg->col_idx[tmp] < 0)) {
					printf("ERROR  VERTEX NUMBER EXECTED!!!\n");
					*ierr = ZOLTAN_FATAL;
					return;
				}
			}
		}

		for (i = 0; i < hg->rows; i++) {
			for (int tmp = hg->row_ptr[i] + 1; tmp < hg->row_ptr[i + 1]; tmp++) {
				if (hg->col_idx[tmp - 1] >= hg->col_idx[tmp]) {
					printf("ERROR  row%d tmp%d  %d < %d\n", i, tmp, hg->col_idx[tmp - 1], hg->col_idx[tmp]);
					*ierr = ZOLTAN_FATAL;
					return;
				}
			}

		}

		if (hg->row_ptr[hg->rows] != hg->nnz) {
			printf("WRONG NNZ!!!");
			*ierr = ZOLTAN_FATAL;
			return;
		}

		for (i = 0; i < hg->rows; i++) {
			edgeGID[i] = i + hg->m_parts[hg->my_rank];
			//			printf("EDGE GLOBAL ID %d\n", edgeGID[i]);
			vtxPtr[i] = hg->row_ptr[i];
		}
		for (i = 0; i < hg->nnz; i++) {
			vtxGID[i] = hg->col_idx[i];
		}


		std::cout << "FINISHED OK" << std::endl;
	}

};
