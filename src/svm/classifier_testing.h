/**
 * Test given classifier on testing dataset
 * @param classifier
 * @param cscValA
 * @param cscRowIndA row indexes
 * @param cscColPtrA
 * @param test_data_label labels
 */
float testClassifierForSVM(std::vector<float> classifier,
		std::vector<float> cscValA,
		std::vector<int> cscRowIndA,
		std::vector<int> cscColPtrA,
		std::vector<float> test_data_label) {
	std::vector<float> predictedLabel(test_data_label.size(), 0);
	for (int col = 0; col < cscColPtrA.size() - 1; col++) {
		for (int i = cscColPtrA[col]; i < cscColPtrA[col + 1]; i++) {
			predictedLabel[cscRowIndA[i]] += cscValA[i]
					* classifier[col];
		}
	}
	int successful = 0;
	for (int i = 0; i < predictedLabel.size(); i++) {

//if (i<10){
//	printf("pred %f, actial %f\n",predictedLabel[i],test_data_label[i]);
//}


		if (sgn(predictedLabel[i]) == sgn(test_data_label[i])) {
			successful++;
		}
	}
	return (float) successful / test_data_label.size();
}

