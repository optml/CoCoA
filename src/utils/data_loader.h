/*
 * This contains functions to load data from CSV file into full matrix representation
 *
 * loadDataFromCSVFile - output matrix is stored in row oriented vector
 *
 */

#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

#include<iostream>
#include<fstream>

using namespace std;

/**
 * Parses data from file
 *
 * @param m - number of rows of matrix stored in CSV file
 * @param n - number of columns of matrix stored in CSV file
 * @param filename - filename of file to be read from
 * @param imageInRowFormat - output vector where matrix will be stored in row-wise format
 */
template<  typename D>
int loadDataFromCSVFile(int m, int n, const char* filename,
		std::vector<D> & data) {
	data.resize(n * m);
	FILE* file = fopen(filename, "r");
	if (file == 0) {
		printf("File '%s'not found\n", filename);
		return 0;
	}
	int nnz = 0;
	char* stringBuffer = (char*) malloc(65536);
	for (int i = 0; i < m; i++) {
		char c;
		char* bufferPointer = stringBuffer;
		do {
			c = fgetc(file);
			if ((c == ' ') || (c == '\n')) {
				//Feature found
				*(bufferPointer) = 0;
				float value;
				sscanf(stringBuffer, "%f", &value);
				data[nnz] = value;
				nnz++;
				bufferPointer = stringBuffer;
			} else if (c == ',') {
				//Position found
				*(bufferPointer) = 0;
				float value;
				sscanf(stringBuffer, "%f", &value);
				data[nnz] = value;
				nnz++;
				bufferPointer = stringBuffer;
			} else {
				*(bufferPointer) = c;
				bufferPointer++;
			}
		} while (c != '\n');
	}
	free(stringBuffer);
	fclose(file);

	return 1;
}

template<typename L, typename D>
int load_matrix_in_coo_format(const char* filename,
		std::vector<D> &A_coo_values, //
		std::vector<L> &A_coo_row_idx,//
		std::vector<L> &A_coo_col_idx,//
		L &m, L &n, bool oneBase=true) {
L subtract = 0;
	if (oneBase){
		subtract=1;
	}


	//	FILE* file = fopen(filename, "r");
	//	if (file == 0) {
	//		printf("File '%s'not found\n", filename);
	//		return 0;
	//	}
	ifstream myReadFile;
	myReadFile.open(filename);
	L x, y;
	D value;

	int nnz = 0;
	m = 0;
	n = 0;
	if (myReadFile.is_open()) {
		//		cout << "Opened " << inname << " for reading." << endl;
		while (myReadFile >> x) {
			nnz++;
			myReadFile >> y;
			myReadFile >> value;
			if (x > m)
				m = x;
			if (y > n)
				n = y;
		}
	} else {
		return 0;
	}
	A_coo_col_idx.resize(nnz);
	A_coo_row_idx.resize(nnz);
	A_coo_values.resize(nnz);
	myReadFile.clear();
	myReadFile.seekg(0);

	nnz = 0;
	if (myReadFile.is_open()) {
		while (myReadFile >> x) {
			myReadFile >> y;
			myReadFile >> value;
			A_coo_col_idx[nnz] = y - subtract;
			A_coo_row_idx[nnz] = x - subtract;
			A_coo_values[nnz] = value;
			nnz++;
		}
	} else {
		return 0;
	}
	myReadFile.close();

	if (!oneBase){
		m=m+1;
		n=n+1;
	}


	return 1;
}

template<typename L, typename D>
int load_matrix_in_csc_format(const char* filename, const char* filename_col,
		std::vector<D> &A_csc_values, //
		std::vector<L> &A_csc_row_idx,//
		std::vector<L> &A_csc_col_ptr,//
		L &m, L &n) {

	ifstream myReadFile;
	myReadFile.open(filename);
	L row_id;
	D value;

	int nnz = 0;
	m = 0;
	n = 0;
	if (myReadFile.is_open()) {
		while (myReadFile >> row_id) {
			nnz++;
			myReadFile >> value;
			if (row_id > m)
				m = row_id;
		}
	} else {
		return 0;
	}
	A_csc_row_idx.resize(nnz);
	A_csc_values.resize(nnz);
	myReadFile.clear();
	myReadFile.seekg(0);

	nnz = 0;
	if (myReadFile.is_open()) {
		while (myReadFile >> row_id) {
			myReadFile >> value;
			A_csc_row_idx[nnz] = row_id;
			A_csc_values[nnz] = value;
			nnz++;
		}
	} else {
		return 0;
	}
	myReadFile.close();

	myReadFile.open(filename_col);
	L col_id;

	nnz = 0;
	if (myReadFile.is_open()) {
		while (myReadFile >> col_id) {
			nnz++;
		}
	} else {
		return 0;
	}
	A_csc_col_ptr.resize(nnz);
	n = nnz - 1;
	myReadFile.clear();
	myReadFile.seekg(0);

	nnz = 0;
	if (myReadFile.is_open()) {
		while (myReadFile >> col_id) {
			A_csc_col_ptr[nnz] = col_id;
			nnz++;
		}
	} else {
		return 0;
	}
	myReadFile.close();
	return 1;
}

#endif /* DATA_LOADER_H_ */
