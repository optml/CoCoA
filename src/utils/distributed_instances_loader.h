/*
 * distributed_instances_loader.h
 *
 *  Created on: May 21, 2013
 *      Author: taki
 */

#ifndef DISTRIBUTED_INSTANCES_LOADER_H_
#define DISTRIBUTED_INSTANCES_LOADER_H_

//#include "../helpers/matrix_conversions.h"
//#include "../problem_generator/distributed/generator_nesterov_to_file.h"
#include "../svm/svm_parser.h"
#include "../solver/structures.h"

string getFileName(string prefix, string type, int file, int files,
		bool local) {
	stringstream ss;
	ss << prefix << "_" << type << "_" << file << "_" << files;
	if (local)
		ss << "_local";
	else
		ss << "_global";
	ss << ".data";
	return ss.str();

}

string getFileName(string prefix, string type, int file, bool local) {
	stringstream ss;
	ss << prefix << "_" << type << "_" << file;
	if (local)
		ss << "_local";
	else
		ss << "_global";
	ss << ".data";
	return ss.str();

}

template<typename D, typename L>
void storeCOOMatrixData(string inputFile, int f, int files,
		ProblemData<L, D> & part) {
	ofstream A_csc_vals;
	ofstream A_csc_row_idx;
	ofstream A_csc_col_ptr;
	A_csc_row_idx.open(getFileName(inputFile, "rowid", f, files, true).c_str(),
			ios::out | ios::binary);
	A_csc_col_ptr.open(getFileName(inputFile, "colptr", f, files, true).c_str(),
			ios::out | ios::binary);
	A_csc_vals.open(getFileName(inputFile, "values", f, files, true).c_str(),
			ios::out | ios::binary);

	for (int i = 0; i < part.A_csc_values.size(); i++) {
		A_csc_vals.write((char*) &part.A_csc_values[i], sizeof(D));
	}
	for (int i = 0; i < part.A_csc_row_idx.size(); i++)
		A_csc_row_idx.write((char*) &part.A_csc_row_idx[i], sizeof(L));
	for (int i = 0; i < part.A_csc_col_ptr.size(); i++) {
		A_csc_col_ptr.write((char*) &part.A_csc_col_ptr[i], sizeof(L));
	}
	A_csc_vals.close();
	A_csc_row_idx.close();
	A_csc_col_ptr.close();

}

template<typename D, typename L>
int loadDistributedSparseSVMRowData(string inputFile, int file, int totalFiles,
		ProblemData<L, D> & part, bool zeroBased) {

	int nclasses;
	int nsamples;
	int nfeatures;
	long long nonzero_elements_of_input_data;

	stringstream ss;
	ss << inputFile;
	if (totalFiles > 0) {
		ss << "." << totalFiles << "." << file;
	}
	cout << "Going to parse SVM data" << endl;
	parse_LIB_SVM_data_get_size(ss.str().c_str(), nsamples, nfeatures,
			nonzero_elements_of_input_data);
	cout << "Data file " << file << " contains " << nfeatures << " features, "
			<< nsamples << " samples " << "and total "
			<< nonzero_elements_of_input_data << " nnz elements" << endl;

	FILE* filePtr = fopen(ss.str().c_str(), "r");
	if (filePtr == 0) {
		printf("File   '%s' not found\n", ss.str().c_str());
		return 0;
	}

	cout << "Going to process data" << endl;

	part.m = nfeatures;
	part.n = nsamples;

	cout << "resize nsamples+1 " << nsamples + 1 << endl;
	part.A_csr_row_ptr.resize(nsamples + 1);
	part.A_csr_col_idx.resize(nonzero_elements_of_input_data);
	cout << "resize nnz " << nonzero_elements_of_input_data << endl;
	part.A_csr_values.resize(nonzero_elements_of_input_data);
	cout << "resize nnz " << nonzero_elements_of_input_data << endl;
	part.b.resize(nsamples);
	cout << "resize nsamples " << nsamples << endl;
	L nnzPossition = 0;
	L processedSamples = -1;


	bool foundData =false;
	char* stringBuffer = (char*) malloc(65536);
	for (L i = 0; i < nsamples; i++) {


		char c;
		L pos = 0;
		char* bufferPointer = stringBuffer;

		do {
			c = fgetc(filePtr);



			//
			if ((c == ' ') || (c == '\n')) {
				if (pos == 0) {
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%i", &value);

					D ddval = value;
					if (value < 100) {
						if (nclasses == 2 && value == 0) {
							ddval = (float) -1;
						} else {
						}

						processedSamples++;
						part.b[processedSamples] = ddval; // used for a1a data
						//part.b[processedSamples] = (-1.5 + ddval) * 2.0; // used for covtype data
						part.A_csr_row_ptr[processedSamples] = nnzPossition;

						pos++;
					}
				} else  {
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);

					if (pos > 0) {

						if (!zeroBased)
							pos--;

						if (nnzPossition < nonzero_elements_of_input_data && foundData) {
							part.A_csr_col_idx[nnzPossition] = pos;
							part.A_csr_values[nnzPossition] = value;

							foundData=false;
							nnzPossition++;
						}

						pos = -1;
					}

				}
				bufferPointer = stringBuffer;
			} else if (c == ':') {
				//Position found
				*(bufferPointer) = 0;
				int value;
				sscanf(stringBuffer, "%i", &value);
				foundData=true;
				pos = value;
				bufferPointer = stringBuffer;
			} else {
				*(bufferPointer) = c;
				bufferPointer++;
			}

		} while (c != '\n');
	}

	processedSamples++;
	part.A_csr_row_ptr[processedSamples] = nnzPossition;
	free(stringBuffer);
	fclose(filePtr);
	return 1;
}

template<typename D, typename L>
void loadDistributedSVMData(string inputFile, int file, int totalFiles,
		ProblemData<L, D> & part) {

	ifstream problemDescription;
	problemDescription.open(
			getFileName(inputFile, "problem", -1, totalFiles, true).c_str());

	L nsamples;
	L nfeatures;

	problemDescription >> nsamples;
	problemDescription >> nfeatures;

	L localN;
	L localNNZ;
	for (int f = 0; f <= file; f++) {
		problemDescription >> localN;
		problemDescription >> localNNZ;
	}
	problemDescription.close();
//	ifstream CooData;
//			CooData.open(getFileName(inputFile, "A_COO", f, true).c_str(),
//					ios::in | ios::binary);
	cout << "Local file loader sizes loaded" << endl;
	ifstream A_csc_vals;
	ifstream A_csc_row_idx;
	ifstream A_csc_col_ptr;
	A_csc_row_idx.open(
			getFileName(inputFile, "rowid", file, totalFiles, true).c_str(),
			ios::in | ios::binary);
	A_csc_col_ptr.open(
			getFileName(inputFile, "colptr", file, totalFiles, true).c_str(),
			ios::in | ios::binary);
	A_csc_vals.open(
			getFileName(inputFile, "values", file, totalFiles, true).c_str(),
			ios::in | ios::binary);

	part.m = nsamples;
	part.n = localN;

	part.b.resize(nsamples);

	ifstream b;
	b.open(getFileName(inputFile, "y", -1, -1, true).c_str(),
			ios::in | ios::binary);
	for (L i = 0; i < nsamples; i++) {
		D tmpValue;
		b.read((char*) &tmpValue, sizeof(D));
		part.b[i] = tmpValue;
	}
	b.close();

	part.A_csc_col_ptr.resize(localN + 1);
	for (L nl = 0; nl < localN + 1; nl++) { //local column indexes
		L tmpValue;
		A_csc_col_ptr.read((char*) &tmpValue, sizeof(L));
		part.A_csc_col_ptr[nl] = tmpValue;
	}

	part.A_csc_row_idx.resize(localNNZ);
	part.A_csc_values.resize(localNNZ);
	for (L nl = 0; nl < localNNZ; nl++) { //local column indexes
		D tmpValueFL;
		L tmpValueI;
		A_csc_row_idx.read((char*) &tmpValueI, sizeof(L));
		A_csc_vals.read((char*) &tmpValueFL, sizeof(D));
		part.A_csc_row_idx[nl] = tmpValueI;
		part.A_csc_values[nl] = tmpValueFL;
	}
	A_csc_vals.close();
	A_csc_row_idx.close();
	A_csc_col_ptr.close();

}

template<typename D, typename L>
D loadDataFromFiles(string logFileName, int file, int totalFiles,
		ProblemData<L, D>& part_global, ProblemData<L, D> &part_local) {

	L local_n;
	L local_m;
	L global_m;
	D optimal_objective;
	ifstream info;
	info.open(logFileName.c_str());
	info >> local_n;
	info >> local_m;
	info >> global_m;
	info >> optimal_objective;
	info.close();

	part_global.m = global_m;
	part_global.n = local_n;

	part_local.m = local_m;
	part_local.n = local_n;

	part_global.b.resize(global_m);
	part_local.b.resize(local_m);
	ifstream b_local;
	ifstream b_global;
	b_local.open(getFileName(logFileName, "b", file, true).c_str());
	b_global.open(getFileName(logFileName, "b", file, false).c_str());

	D tmpValue;
	for (L i = 0; i < local_m; i++) {
		b_local >> tmpValue;
		part_local.b[i] = tmpValue;
	}
	for (L i = 0; i < global_m; i++) {
		b_global >> tmpValue;
		part_global.b[i] = tmpValue;
	}
	b_local.close();
	b_global.close();

	ifstream A_csc_local_col_ptr;
	ifstream A_csc_local_vals_INPUT;
	ifstream A_csc_global_col_ptr;
	ifstream A_csc_global_vals_INPUT;
	ifstream A_csc_local_row_id;
	ifstream A_csc_global_row_id;
	A_csc_local_row_id.open(
			getFileName(logFileName, "rowid", file, true).c_str(),
			ios::out | ios::binary);
	A_csc_global_row_id.open(
			getFileName(logFileName, "rowid", file, false).c_str(),
			ios::out | ios::binary);
	A_csc_local_col_ptr.open(
			getFileName(logFileName, "colptr", file, true).c_str());
	A_csc_local_vals_INPUT.open(
			getFileName(logFileName, "values", file, true).c_str(),
			ios::out | ios::binary);
	A_csc_global_col_ptr.open(
			getFileName(logFileName, "colptr", file, false).c_str());
	A_csc_global_vals_INPUT.open(
			getFileName(logFileName, "values", file, false).c_str(),
			ios::out | ios::binary);

	L localPtr;
	L globalPtr;
	A_csc_local_col_ptr >> localPtr;
	A_csc_global_col_ptr >> globalPtr;
	part_global.A_csc_col_ptr.resize(local_n + 1);
	part_local.A_csc_col_ptr.resize(local_n + 1);
	part_global.A_csc_col_ptr[0] = globalPtr;
	part_local.A_csc_col_ptr[0] = localPtr;
	for (L nl = 0; nl < local_n; nl++) { //local column indexes
		A_csc_local_col_ptr >> localPtr;
		A_csc_global_col_ptr >> globalPtr;
		part_global.A_csc_col_ptr[1 + nl] = globalPtr;
		part_local.A_csc_col_ptr[1 + nl] = localPtr;
	}
	part_global.A_csc_row_idx.resize(globalPtr);
	part_global.A_csc_values.resize(globalPtr);
	part_local.A_csc_row_idx.resize(localPtr);
	part_local.A_csc_values.resize(localPtr);

	L IDX;
	D value;
	for (L i = 0; i < localPtr; i++) { //local column indexes
//		A_csc_local_row_id >> IDX;
//		A_csc_local_vals_INPUT >> value;
		A_csc_local_row_id.read((char*) &IDX, sizeof(L));
		A_csc_local_vals_INPUT.read((char*) &value, sizeof(D));
		part_local.A_csc_row_idx[i] = IDX;
		part_local.A_csc_values[i] = value;
	}
	for (int i = 0; i < globalPtr; i++) {
//		A_csc_global_row_id >> IDX;
//		A_csc_global_vals_INPUT >> value;
		A_csc_global_row_id.read((char*) &IDX, sizeof(L));
		A_csc_global_vals_INPUT.read((char*) &value, sizeof(D));

		part_global.A_csc_row_idx[i] = IDX;
		part_global.A_csc_values[i] = value;
	}
	A_csc_local_col_ptr.close();
	A_csc_local_vals_INPUT.close();
	A_csc_global_col_ptr.close();
	A_csc_global_vals_INPUT.close();
	A_csc_local_row_id.close();
	A_csc_global_row_id.close();
	return optimal_objective;
}

#endif /* DISTRIBUTED_INSTANCES_LOADER_H_ */
