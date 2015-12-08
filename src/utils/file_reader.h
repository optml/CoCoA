/*
 *
 * This is a parallel sparse PCA solver
 *
 * The solver is based on a simple alternating maximization (AM) subroutine 
 * and is based on the paper
 *    P. Richtarik, M. Takac and S. Damla Ahipasaoglu 
 *    "Alternating Maximization: Unifying Framework for 8 Sparse PCA Formulations and Efficient Parallel Codes"
 *
 * The code is available at https://code.google.com/p/24am/
 * under GNU GPL v3 License
 * 
 */

#ifndef FILE_READER_H_
#define FILE_READER_H_

#include <vector>
#include<iostream>
#include<fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <iomanip>
using namespace std;
#include "../Context.h"
#include "../solver/structures.h"
namespace InputOuputHelper {

template<typename G>
void loadVectorData(std::string &fileLocation, std::vector<G>& vector) {
	cout <<"Loading file :"<<fileLocation<<endl;
	std::ifstream myfile(fileLocation.c_str());
	G val;
	vector.resize(0);
	if (myfile.is_open()) {
		myfile >> val;
		while (!myfile.eof()) {
			vector.push_back(val);
			myfile >> val;
		}
		myfile.close();
	}else{
		cout << "File NOT FOUND!!!"<<endl;
	}
}

template<typename L, typename D>
void loadANSCIAndBinaryData(Context & context, std::string matrixPrefix,
		std::string matrixSufix, std::string vectorBFileName,
		ProblemData<L, D> &instance, L local_n, L m, int rank) {

	cout << "Loading " << vectorBFileName << endl;
	InputOuputHelper::loadVectorData(vectorBFileName, instance.b);

	std::stringstream ss;
	ss << matrixPrefix << "_colptr_" << rank << matrixSufix;
	cout << "Loading " << ss.str() << "tpota " << local_n << endl;
	ifstream is;
	is.open(ss.str().c_str(), ios::in | ios::binary);
	instance.A_csc_col_ptr.resize(local_n + 1);
	for (L j = 0; j < local_n + 1; j++) {
		is >> instance.A_csc_col_ptr[j];
	}
	is.close();
	ss.str("");

	ss << matrixPrefix << "_rowid_" << rank << matrixSufix;
	cout << "Loading " << ss.str() << "  " << instance.A_csc_col_ptr[local_n] << endl;
	is.open(ss.str().c_str(), ios::in | ios::binary);
	instance.A_csc_row_idx.resize(instance.A_csc_col_ptr[local_n]);
	for (L j = 0; j < instance.A_csc_col_ptr[local_n]; j++) {
		L val;
		is.read((char*) &val, sizeof(L));
		instance.A_csc_row_idx[j] = val;
	}
	is.close();
	ss.str("");

	ss << matrixPrefix << "_values_" << rank << matrixSufix;
	cout << "Loading " << ss.str() << "tpota " << instance.A_csc_col_ptr[local_n] << endl;
	is.open(ss.str().c_str(), ios::in | ios::binary);
	instance.A_csc_values.resize(instance.A_csc_col_ptr[local_n]);
	for (L j = 0; j < instance.A_csc_col_ptr[local_n]; j++) {
		D val;
		is.read((char*) &val, sizeof(D));
		instance.A_csc_values[j] = val;
	}
	is.close();
	ss.str("");

//	ss << matrixPrefix << "_values.txt";
//	std::string file = ss.str();
//	InputOuputHelper::loadVectorData(file, instance.A_csc_values);
//	ss.str("");
//	ss.str("");
//	ss << matrixPrefix << "_rowIdx.txt";
//	file = ss.str();
//	InputOuputHelper::loadVectorData(file, instance.A_csc_row_idx);

	instance.n = instance.A_csc_col_ptr.size() - 1;
	instance.m = instance.b.size();
	instance.lambda = context.lambda;
}

template<typename L, typename D>
void loadCSCData(Context & context, std::string matrixPrefix,
		std::string vectorBFileName, ProblemData<L, D> &instance) {




	InputOuputHelper::loadVectorData(vectorBFileName, instance.b);
	std::stringstream ss;
	ss << matrixPrefix << "_values.txt";
	std::string file = ss.str();
	InputOuputHelper::loadVectorData(file, instance.A_csc_values);
	ss.str("");
	ss << matrixPrefix << "_colPtr.txt";
	file = ss.str();
	InputOuputHelper::loadVectorData(file, instance.A_csc_col_ptr);
	ss.str("");
	ss << matrixPrefix << "_rowIdx.txt";
	file = ss.str();
	InputOuputHelper::loadVectorData(file, instance.A_csc_row_idx);
	instance.n = instance.A_csc_col_ptr.size() - 1;
	instance.m = instance.b.size();
	instance.lambda = context.lambda;
}

template<typename L, typename D>
void loadCSCData(Context & context, ProblemData<L, D> &instance) {
	InputOuputHelper::loadCSCData(context, context.matrixAFile,
			context.vectorBFile, instance);
}

void parse_data_size_from_CSV_file(unsigned int &m, unsigned int &n,
		const char* input_csv_file) {
	m = 0;
	n = 0;
	std::ifstream data(input_csv_file);
	std::string line;
	while (std::getline(data, line)) {
		if (m == 0) {
			std::stringstream lineStream(line);
			std::string cell;
			while (std::getline(lineStream, cell, ',')) {
				n++;
			}
		}
		m++;
	}
}

/**
 * Parses data from file
 *
 * @param m - number of rows of matrix stored in CSV file
 * @param n - number of columns of matrix stored in CSV file
 * @param filename - filename of file to be read from
 */
template<typename D>
int parse_data_from_CSV_file(unsigned int m, unsigned int n,
		const char* input_csv_file, std::vector<D> & data) {
	data.resize(n * m);
	ifstream my_read_file;
	my_read_file.open(input_csv_file);
	D value;
	if (my_read_file.is_open()) {
		for (unsigned int row = 0; row < m; row++) {
			for (unsigned int col = 0; col < n; col++) {
				my_read_file >> value;
				char c;
				if (col < n - 1)
					my_read_file >> c;
				data[row + col * m] = value;
			}
		}
		my_read_file.close();
		return 0;
	} else {
		return 1;
	}
}

template<typename F>
void readCSVFile(std::vector<F> &Bmat, unsigned int &ldB, unsigned int &m,
		unsigned int & n, const char* input_csv_file) {
	parse_data_size_from_CSV_file(m, n, input_csv_file);
	parse_data_from_CSV_file(m, n, input_csv_file, Bmat);
	ldB = m;
}

}
#endif /* FILE_READER_H_ */
