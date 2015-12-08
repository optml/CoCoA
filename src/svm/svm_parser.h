/*
 * svm_parser.h
 *
 *  Created on: May 21, 2013
 *      Author: taki
 */

#ifndef SVM_PARSER_H_
#define SVM_PARSER_H_

int parse_LIB_SVM_data_get_size(const char* filename, int &nsamples,
		int &nfeatures, long long &nonzero_elements_of_input_data) {
	nfeatures = 0;
	nsamples = 0;
	nonzero_elements_of_input_data = 0;
	FILE* file = fopen(filename, "r");
	if (file == 0) {
		printf("File '%s' not found\n", filename);
		return 0;
	}
	char* stringBuffer = (char*) malloc(65536);
	bool end_of_file = false;
	nsamples = -1;
	//	for (int i = 0; i < nsamples; i++) {
	while (!end_of_file) { //feof(file)
		nsamples++;
		char c;
		int pos = 0;
		char* bufferPointer = stringBuffer;
		do {
			c = fgetc(file);
			if (c == EOF) {
				end_of_file = true;
				break;
			}

			if ((c == ' ') || (c == '\n')) {
				if (pos == 0) {
					//Label found
					*(bufferPointer) = 0;
					int value;
					sscanf(stringBuffer, "%i", &value);

					if (value < 100)
						pos++;

				} else {
					//Feature found
					*(bufferPointer) = 0;
					float value;
					sscanf(stringBuffer, "%f", &value);
					if (pos > 0) {
						if (nfeatures < pos)
							nfeatures = pos;
						pos = -1;
						nonzero_elements_of_input_data++;
					}
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
	return 1;
}

#endif /* SVM_PARSER_H_ */
