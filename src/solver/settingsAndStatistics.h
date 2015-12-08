/*
 * settingsAndStatistics.h
 *
 *  Created on: Jun 27, 2013
 *      Author: taki
 */

#ifndef SETTINGSANDSTATISTICS_H_
#define SETTINGSANDSTATISTICS_H_

class OptimizationSettings {
public:

	int lossFunction;
	int LocalMethods;
	bool showInitialObjectiveValue;
	bool showIntermediateObjectiveValue;
	bool showLastObjectiveValue;
	bool show_prediction_accuracy;

	double threshold;

	int regulatization_path_length;

	long allocation_treshold_for_multilevel; // from some treshold on, we have to use multilevel

	bool use_atomic;
	int totalThreads;
	int torus_width;
	int broadcast_treshold;
	float supersparse_epsilon;
	bool capture_ethernet_stats;
	bool capture_infiniband_stats;

	double total_execution_time;

	double minimal_target_value;

	bool recomputeResidualAfterEachBulkIteration;

	bool bulkIterations;
	bool verbose;

	bool APPROX;

	bool logToFile;
	std::ofstream* logFile;
	int device_block_dim_1;
	int device_block_dim_2;
	int device_total_threads_per_block;
	int device_memory_block_size;

	double forcedSigma;

	unsigned long iters_bulkIterations_count;
	unsigned long iters_communicate_count;
	unsigned long innerIterations;

	// Shrinking for SR
	bool use_shrinking;
	double shrinking_probability;
	int shrinking_starting_iter;

	bool block_random_coodinates;

	bool use_double_uniform_sampling;

	// GPU optimization of PR solver
	bool aligned_memory_access;
	bool permute_data;

	OptimizationSettings() : //Set default values to settings
			iters_bulkIterations_count(10), iters_communicate_count(100), innerIterations(
					100), device_block_dim_1(14), device_block_dim_2(1), device_total_threads_per_block(
					64), recomputeResidualAfterEachBulkIteration(false), use_double_uniform_sampling(
					true), permute_data(false), device_memory_block_size(32), threshold(
					0), showIntermediateObjectiveValue(true), minimal_target_value(
					0), aligned_memory_access(true), regulatization_path_length(
					0), show_prediction_accuracy(false), showLastObjectiveValue(
					true), block_random_coodinates(true), verbose(false), bulkIterations(
					false), totalThreads(8), showInitialObjectiveValue(false), use_atomic(
					false), forcedSigma(-1), use_shrinking(false), logToFile(
					false), allocation_treshold_for_multilevel(
					1024 * 1024 * 1024), torus_width(1), broadcast_treshold(20), supersparse_epsilon(
					0), capture_ethernet_stats(false), APPROX(false), capture_infiniband_stats(
					true) {
//		struct sysinfo memInfo;
//		sysinfo(&memInfo);
//		allocation_treshold_for_multilevel = memInfo.freeram
//				- 100 * 1024 * 1024;

	}
};

class OptimizationStatistics {

public:
	double last_obj_value;
	float prediction_accuracy;

	double time_wallclock;

	int time_rounds; // TODO: Change to L or long long
	unsigned long long time_iterations;

	float number_of_iters_in_millions;
	float average_speed_iters_per_ms;

	std::string instance_name;
	long instance_nnz;
	long instance_allocates;
	long instance_columns;
	long instance_rows;

	int parts;
	int hypergraph_cut;
	double imbalance_columns;
	double imbalance_nnz;
	int max_nnz_per_row;
	int max_nnz_per_col;
	long floats_exchanged_per_comm_iter;
	int residual_rows_with_reduce;
	int max_number_of_partitions_involved_in_one_row;

	long vm_usage_local;
	long resident_set_local;

	long vm_usage_total;
	long resident_set_total;

	long long received_bytes_local;
	long long transmitted_bytes_local;
	double traffic_mbytes_local;

	long long received_bytes_total;
	long long transmitted_bytes_total;
	double traffic_mbytes_total;

	double elapsed_time;
	double elapsedPureComputationTime;
	unsigned long elapsedIterations;

	OptimizationStatistics() {
		elapsedIterations = 0;
		elapsedPureComputationTime = 0;
		elapsed_time = 0;
	}

};

#endif /* SETTINGSANDSTATISTICS_H_ */
