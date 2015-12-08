/**
 * Parses data from file
 * Data will be stored in CSC zero‐based storage format
 *
 * @param filename
 * @param h_data_label labels
 * @param nsamples number of samples in the training set
 * @param nfeatures number of features per sample in the training set
 * @param nclasses number of classes
 * @param nonzero_elements_of_input_data maximum nonzero elements of data
 *
 *
 *
 */
#ifndef PARSE_INPUT_HPP
#define PARSE_INPUT_HPP

#include "../helpers/matrix_conversions.h"

#include "svm_parser.h"

using namespace std;
// namespace for std;

template<typename L, typename D>
int parseLibSVMdata(const char* filename, std::vector<D>& cscValA,
		std::vector<L>& cscRowIndA, std::vector<L>& cscColPtrA,
		std::vector<D>& h_data_label, int nsamples, int nfeatures, int nclasses,
		long long nonzero_elements_of_input_data) {
	//	long long maxreqited=nsamples * nfeatures;
	//	if (nonzero_elements_of_input_data > maxreqited) {
	//		//maximum nonzero elements can be  nsamples * nfeatures
	//		nonzero_elements_of_input_data = maxreqited;
	//	}
	std::vector<D> h_data_values(nonzero_elements_of_input_data);
	std::vector<L> h_data_row_index(nonzero_elements_of_input_data);
	std::vector<L> h_data_col_index(nonzero_elements_of_input_data);
	std::vector<L> colCount(nfeatures, 0);
	(h_data_label).resize(nsamples);
	int nnz = 0;
	FILE* file = fopen(filename, "r");
	if (file == 0) {
		printf("File '%s' not found\n", filename);
		return 0;
	}
	char* stringBuffer = (char*) malloc(65536);
	for (int i = 0; i < nsamples; i++) {
		char c;
		int pos = 0;
		char* bufferPointer = stringBuffer;
		do {
			c = fgetc(file);

			if ((c == ' ') || (c == '\n')) {
				if (pos == 0) {
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%i", &value);

					if (value < 100) {
						//						printf("value=%d\n", value);
						if (nclasses == 2 && value == 0) {
							(h_data_label)[i] = (float) -1;
						} else {
							(h_data_label)[i] = (float) value;
						}
						pos++;

						//						printf("Label Found: %f row %d \n", (h_data_label)[i],
						//								i);
					}
				} else {
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);

					h_data_values[nnz] = value;
					h_data_row_index[nnz] = i + 1;
					h_data_col_index[nnz] = pos;
					colCount[pos - 1]++;
					pos = 0;

					//					printf("Feautre Found: X[%d , %d]=%f \n",
					//							h_data_row_index[nnz], h_data_col_index[nnz],
					//							h_data_values[nnz]);
					nnz++;

				}
				bufferPointer = stringBuffer;
			} else if (c == ':') {
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				pos = value;
				bufferPointer = stringBuffer;
			} else {
				*(bufferPointer) = c;
				bufferPointer++;
			}

		} while (c != '\n');
	}
	free(stringBuffer);
	fclose(file);
	for (L i = 0; i < nnz; i++) {
		h_data_row_index[i]--;
		h_data_col_index[i]--;
	}
	getCSC_from_COO(h_data_values, h_data_row_index, h_data_col_index, cscValA,
			cscRowIndA, cscColPtrA, nsamples, nfeatures);

	return 1;
}

template<typename L, typename D>
int parse_lib_SVM_data_into_CSR(const char* filename, std::vector<D>& cscValA,
		std::vector<L>& cscColIndA, std::vector<L>& cscRowPtrA,
		std::vector<D>& h_data_label, int nsamples, int nfeatures, int nclasses,
		long long nonzero_elements_of_input_data, mpi::communicator &world) {
	//	long long maxreqited=nsamples * nfeatures;
	//	if (nonzero_elements_of_input_data > maxreqited) {
	//		//maximum nonzero elements can be  nsamples * nfeatures
	//		nonzero_elements_of_input_data = maxreqited;
	//	}
//	nonzero_elements_of_input_data=nonzero_elements_of_input_data;
	std::vector<D> h_data_values(nonzero_elements_of_input_data);
//	world.barrier();
	std::vector<L> h_data_row_index(nonzero_elements_of_input_data);
//	world.barrier();
	std::vector<L> h_data_col_index(nonzero_elements_of_input_data);
//	world.barrier();
	std::vector<L> colCount(nfeatures, 0);
	(h_data_label).resize(nsamples);
//	cout << "SAMPLES " << world.rank() << " " << nonzero_elements_of_input_data
//			<< " " << getVirtualMemoryCurrentlyUsedByCurrentProcess() << " "
//			<< getPhysicalMemoryCurrentlyUsedByCurrentProcess() << " "
//			<< getTotalSystemMemory() << endl;
	world.barrier();
	int nnz = 0;
	FILE* file = fopen(filename, "r");
	if (file == 0) {
		printf("File '%s' not found\n", filename);
		return 0;
	}
	char* stringBuffer = (char*) malloc(65536);
	for (int i = 0; i < nsamples; i++) {
		char c;
		int pos = 0;
		char* bufferPointer = stringBuffer;
		do {
			c = fgetc(file);

			if ((c == ' ') || (c == '\n')) {
				if (pos == 0) {
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%i", &value);

					if (value < 100) {
						//						printf("value=%d\n", value);
						if (nclasses == 2 && value == 0) {
							(h_data_label)[i] = (float) -1;
						} else {
							(h_data_label)[i] = (float) value;
						}
						pos++;

						//						printf("Label Found: %f row %d \n", (h_data_label)[i],
						//								i);
					}
				} else {
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);

					h_data_values[nnz] = value;
					h_data_row_index[nnz] = i + 1;
					h_data_col_index[nnz] = pos;
					colCount[pos - 1]++;
					pos = 0;

					//					printf("Feautre Found: X[%d , %d]=%f \n",
					//							h_data_row_index[nnz], h_data_col_index[nnz],
					//							h_data_values[nnz]);
					nnz++;

				}
				bufferPointer = stringBuffer;
			} else if (c == ':') {
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				pos = value;
				bufferPointer = stringBuffer;
			} else {
				*(bufferPointer) = c;
				bufferPointer++;
			}

		} while (c != '\n');
	}
	free(stringBuffer);
	fclose(file);
	for (L i = 0; i < nnz; i++) {
		h_data_row_index[i]--;
		h_data_col_index[i]--;
	}
	cout << "GOING TO DO TRANSFORMATION " << nonzero_elements_of_input_data
			<< endl;
	getCSR_from_COO(h_data_values, h_data_row_index, h_data_col_index, cscValA,
			cscColIndA, cscRowPtrA, nsamples, nfeatures);

	return 1;
}


#endif
