/*
 * This contains functions to store data to CSV file from full matrix representation
 *
 * saveDataToCSVFile - matrix is stored into file with name "filename"
 *
 */

#ifndef CSV_WRITTER_H_
#define CSV_WRITTER_H_

/**
 * Parses data from file
 *
 * @param m - number of rows of matrix
 * @param n - number of columns of matrix
 * @param filename - filename of file to be write to
 * @param imageInRowFormat - vector where matrix is stored in row-wise format
 */
template<typename T>
void saveDataToCSVFile(int m, int n, const char* filename, std::vector<T> imageInRowFormat) {
	FILE* file = fopen(filename, "w");
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < (n - 1); col++) {
			fprintf(file, "%1.16f,", imageInRowFormat[row * n + col]);
		}
		fprintf(file, "%1.16f\n", imageInRowFormat[row * n + (n - 1)]);
	}
	fclose(file);
}

template<typename T, typename I>
void save_matrix_in_csc_format(const char* filename, const char* filenamecol, std::vector<T> A_csc_values, std::vector<
		I> A_csc_row_idx, std::vector<I> A_csc_col_ptr) {
	FILE* file = fopen(filename, "w");
	for (I row = 0; row < A_csc_row_idx.size(); row++) {
		fprintf(file, "%d %1.16f\n", A_csc_row_idx[row], A_csc_values[row]);
	}
	fclose(file);

	file = fopen(filenamecol, "w");
	for (I row = 0; row < A_csc_col_ptr.size(); row++) {
		fprintf(file, "%d\n", A_csc_col_ptr[row]);
	}
	fclose(file);
}

template<typename T, typename I>
void save_matrix_in_csc_format_as_coo(const char* filename, std::vector<T> A_csc_values, std::vector<I> A_csc_row_idx,
		std::vector<I> A_csc_col_ptr) {
	FILE* file = fopen(filename, "w");
	for (I col = 0; col < A_csc_col_ptr.size() - 1; col++) {
		for (I row = A_csc_col_ptr[col]; row < A_csc_col_ptr[col+1]; row++) {
			fprintf(file, "%d,%d,%1.16f\n", A_csc_row_idx[row], col,A_csc_values[row]);
		}
	}
	fclose(file);
}

#endif /* CSV_WRITTER_H_ */
