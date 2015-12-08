/*
 * Optimization and output structures,
 *
 * OptimizationSettings structure is used to configure solvers, how much can solver run,
 * if solver should show objective values after "iters_bulkIterations_count"
 * iterations or not and configuration of Kernel
 */

#ifndef DISTRIBUTED_STRUCTURES_H_
#define DISTRIBUTED_STRUCTURES_H_

#include <ios>
#include <cstdlib>
#include <unistd.h>

#include "sys/types.h"
#include "sys/sysinfo.h"

#include <list>

#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <string>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/reduce.hpp>
namespace mpi = boost::mpi;

#include <boost/timer/timer.hpp>
using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;

#include "distributed_asynchronous_topologies.h"

enum DistributedMethod {
	SynchronousGather = 0,
	SynchronousReduce,
	SynchronousPointToPoint,
	SynchronousSparse,
	SynchronousSupersparse,
	AsynchronousBuffered,
	AsynchronousStreamlined,
	AsynchronousStreamlinedV2,
	AsynchronousStreamlinedOptimized,
	AsynchronousTorus,
	AsynchronousTorusOpt,
	AsynchronousTorusOptCollectives,
	AsynchronousTorusCollectives
};

template<typename T> // NOTE: We use typename T instead of std::ostream to make this header-only
T& operator<<(T& stream, DistributedMethod& algo) {
	switch (algo) {
	case SynchronousGather:
		stream << "SynchronousGather";
		break;
	case SynchronousReduce:
		stream << "SynchronousReduce";
		break;
	case SynchronousPointToPoint:
		stream << "SynchronousPointToPoint";
		break;
	case SynchronousSparse:
		stream << "SynchronousSparse";
		break;
	case SynchronousSupersparse:
		stream << "SynchronousSupersparse";
		break;
	case AsynchronousBuffered:
		stream << "AsynchronousBuffered";
		break;
	case AsynchronousStreamlined:
		stream << "AsynchronousStreamlined";
		break;
	case AsynchronousStreamlinedV2:
		stream << "AsynchronousStreamlinedV2";
		break;
	case AsynchronousStreamlinedOptimized:
		stream << "AsynchronousStreamlinedOptimized";
		break;
	case AsynchronousTorus:
		stream << "AsynchronousTorus";
		break;
	case AsynchronousTorusOpt:
		stream << "AsynchronousTorusOpt";
		break;
	case AsynchronousTorusOptCollectives:
		stream << "AsynchronousTorusOptCollectives";
		break;

	case AsynchronousTorusCollectives:
		stream << "AsynchronousTorusCollectives";
		break;
	}
	return stream;
}

enum PartitioningMethod {
	RandomPartitioning = 0,
	BlockedPartitioning,
	ZoltanPartitioning,
	ZoltanMultilevelPartitioning,
};

enum LogFormat {
	RawData = 0, RawHeaders, Verbose,
};

template<typename T>
T& operator<<(T& stream, PartitioningMethod& method) {
	switch (method) {
	case RandomPartitioning:
		stream << "Random";
		break;
	case BlockedPartitioning:
		stream << "Blocked";
		break;
	case ZoltanPartitioning:
		stream << "Zoltan";
		break;
	case ZoltanMultilevelPartitioning:
		stream << "ZoltanMultilevel";
		break;
	}
	return stream;
}

template<typename L, typename D>
class data_distributor {
public:
	void init(L _n, L _samples_count) {
		n = _n;
		samples_count = _samples_count;
		countsPtr.resize(samples_count + 1, 0);
		sigmas.resize(samples_count, 1);
		indexes.resize(n, -1);
	}
	std::vector<L> columns_parts;
	L samples_count;
	L n;
	std::vector<L> countsPtr;
	std::vector<L> indexes;
	std::vector<D> sigmas;

	std::vector<L> global_row_id_mapper;

	std::list< std::list<L> > what_to_exchange;

	std::set<L> talking_members;
	std::vector<L> coordinates_of_reduce;
	std::vector< std::list<L> > coordinates_of_updates;
	std::vector<std::vector<D> > residual_update_buffer_sent;
	std::vector<std::vector<D> > residual_update_buffer_receive;

	std::vector<D> residual_update_buffer_reduce_sent;
	std::vector<D> residual_update_buffer_reduce_receive;

	//	std::list<L> coordinates_talk_to;
	//	std::list<std::list<L> > coordinates_to_whom_exchange;
	//	std::list<L> coordinates_to_broadcast;
	//	std::list<L> broadcasting;
	//	std::list<std::list<L> > what_to_broadcast;

};

#include "../settingsAndStatistics.h"

class DistributedSettings: public OptimizationSettings/* TODO: We may want to have this derived from OptimizationSettingss */{
public:
//	// From OptimizationSettingss
//
//	bool showInitialObjectiveValue;
//	bool showIntermediateObjectiveValue;
//	bool showLastObjectiveValue;
//	bool show_prediction_accuracy;
//
//	// Shrinking for SR
//	bool use_shrinking;
//	double shrinking_probability;
//	int shrinking_starting_iter;
//
//	// Newly-defined
//


	enum DistributedMethod distributed;
//	int totalThreads;
//	int broadcast_treshold;
//	float supersparse_epsilon;
//	bool use_atomic;
//
//	int torus_width;
//	Topology<int, torus1_indexing> topology;
//
	enum PartitioningMethod partitioning;
//	long allocation_treshold_for_multilevel; // from some treshold on, we have to use multilevel
//
//	bool capture_ethernet_stats;
//	bool capture_infiniband_stats;
//
//	bool bulkIterations;
//	bool verbose;
//
//	double minimal_target_value;
//	unsigned long iters_bulkIterations_count;
//	unsigned long iters_communicate_count;
	unsigned long iterationsPerThread;

	DistributedSettings()   //Set default values to settings
//			iters_bulkIterations_count(100000), iters_communicate_count(
//					100), iterationsPerThread(10), showIntermediateObjectiveValue(
//					true), minimal_target_value(0), show_prediction_accuracy(
//					false), showLastObjectiveValue(true), verbose(false), bulkIterations(
//					false), totalThreads(1), showInitialObjectiveValue(
//					false), use_atomic(false), use_shrinking(false), partitioning(
//					ZoltanPartitioning), allocation_treshold_for_multilevel(
//					1024 * 1024 * 1024), distributed(SynchronousGather), torus_width(
//					1), broadcast_treshold(20), supersparse_epsilon(0), capture_ethernet_stats(
//					true), capture_infiniband_stats(true)
	{
		this->verbose = true;
		this->iters_bulkIterations_count = 5;
		this->iters_communicate_count = 1000;
		iterationsPerThread = 1;
		this->bulkIterations=1;
//		struct sysinfo memInfo;
//		if (sysinfo(&memInfo) == 0 && memInfo.freeram > 100 * 1024 * 1024) {
//			std::cout << "NOTE: sysinfo reports there is " << memInfo.freeram
//					<< " free RAM." << std::endl;
//			allocation_treshold_for_multilevel = memInfo.freeram
//					- 100 * 1024 * 1024;
//		}
//
	}
};

// TODO: We may want this to be derived from OptimizationStatistics (?)
struct distributed_statistics {

	double generated_optimal_value;
	double sigma;
	double last_obj_value;
	float prediction_accuracy;
	int total_mpi_processes;
	int time_rounds; // TODO: Change to L or long long
	unsigned long long time_iterations;

	//	boost::posix_time::ptime t1

	boost::timer::cpu_timer time_nonstop;
	boost::timer::cpu_timer time_pure; //pure conputation (LOOSES updates)
	boost::timer::cpu_timer time_initialization; //time when residuals, Lipschitz constains and other are computed
	boost::timer::cpu_timer time_all_computation; //start at beggining of outer loop and finish at the end
	boost::timer::cpu_timer time_residual_exchange; //measure only echange of redisuals
	boost::timer::cpu_timer time_sample_coordinates; //measure only time to preapre coorinates
	boost::timer::cpu_timer time_other_overhed; //measure time which is spent on other stuff

	boost::timer::nanosecond_type time_pure_total;
	boost::timer::nanosecond_type time_user_total;

	float number_of_iters_in_millions;
	float average_speed_iters_per_ms;

	std::string instance_name;
	long instance_nnz;
	long instance_allocates;
	long instance_columns;
	long instance_rows;

	long m_total;
	long n_total;

	int parts;
	long hypergraph_cut;
	double imbalance_columns;
	double imbalance_nnz;
	int max_nnz_per_row;
	int max_nnz_per_col;
	long floats_exchanged_per_comm_iter;
	int residual_rows_with_reduce;
	long max_parts_per_row;

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

protected:
	long long received_bytes_initial;
	long long transmitted_bytes_initial;

	DistributedSettings settings;
	mpi::communicator *world;
	std::ostream &log_progress;

public:

	distributed_statistics(const char *instance,
			mpi::communicator *world_communicator,
			DistributedSettings current_setting,
			std::ostream &log_progress_stream = std::cout) :
			instance_name(instance), world(world_communicator), settings(
					current_setting), log_progress(log_progress_stream) {
		total_mpi_processes = world->size();
		reset();
	}

	distributed_statistics() :
			log_progress(std::cout) {
		distributed_statistics("Not Set", NULL, DistributedSettings(),
				std::cout);
		// std::cerr << "NOTE: Statistics do not include any details of the settings." << std::endl;
	}

	void reset() {

		last_obj_value = 0;
		prediction_accuracy = 0;

		time_rounds = 1;
		time_iterations = 1;
		number_of_iters_in_millions = 0;
		average_speed_iters_per_ms = 0;

		time_nonstop.stop();
		time_nonstop.start();
		time_nonstop.stop();
		time_pure.stop(); // started only in the loop
		time_pure.start();
		time_pure.stop();
		time_pure_total = 0;
		time_user_total = 0;

		// NOT: CANNOT do instance_name = "Not set";
		instance_nnz = 0;
		instance_allocates = 0;
		instance_columns = 0;
		instance_rows = 0;

		if (world != NULL)
			parts = world->size();
		hypergraph_cut = 0;
		imbalance_columns = 0;
		imbalance_nnz = 0;
		max_nnz_per_row = 0;
		max_nnz_per_col = 0;
		floats_exchanged_per_comm_iter = 0;
		residual_rows_with_reduce = 0;
		max_parts_per_row = 0;
		vm_usage_local = 0;
		resident_set_local = 0;
		vm_usage_total = 0;
		resident_set_total = 0;
		received_bytes_local = 0;
		transmitted_bytes_local = 0;
		traffic_mbytes_local = 0;
		received_bytes_total = 0;
		transmitted_bytes_total = 0;
		traffic_mbytes_total = 0;

		capture_usage(true);
		if (world != NULL and world->rank() == 0) {
			print_results(log_progress, RawHeaders);
			log_progress << std::endl;
		}
	}

	void print_results_to_log() {
		if (world != NULL and world->rank() == 0) {
			print_results(log_progress, RawData);
			log_progress << std::endl;
		}
	}

	std::ostream& print_results(std::ostream& stream,
			LogFormat format = RawData) {
		if (world != NULL and world->rank() != 0)
			return stream;

		switch (format) {

		case RawHeaders:

			// PLEASE KEEP RawData in sync!
			stream << "rounds" << "\t" << "obj" << "\t";
			if (settings.show_prediction_accuracy)
				stream << "accuracy" << "\t";
			stream << "walltime [s]" << "\t" << "compute  [s]" << "\t";
			if (world != NULL)
				stream << "total user  [s]" << "\t" << "total compute [s]"
						<< "\t";
			stream << "Miters" << "\t" << "iters/ms" << "\t";
			stream << "VM" << "\t" << "RSS";
			if (settings.capture_infiniband_stats
					|| settings.capture_ethernet_stats)
				stream << "\t" << "Rx (B)" << "\t" << "Tx (B)" << "\t"
						<< "Traffic (MB)";
			if (world != NULL) {
				stream << "\t" << "total VM" << "\t" << "total RSS";
				if (settings.capture_infiniband_stats
						|| settings.capture_ethernet_stats)
					stream << "\t" << "total Rx (B)" << "\t" << "total Tx (B)"
							<< "\t" << "total traffic (MB)";
			}

			break;

		case RawData:

			// PLEASE KEEP RawHeaders in sync!
			stream << time_rounds << "\t" << last_obj_value << "\t";
			if (settings.show_prediction_accuracy)
				stream << prediction_accuracy << "\t";
			stream << 1.0e-9 * time_nonstop.elapsed().wall << "\t"
					<< 1.0e-9 * time_pure.elapsed().user << "\t";
			if (world != NULL)
				stream << 1.0e-9 * time_user_total << "\t"
						<< 1.0e-9 * time_pure_total << "\t";
			stream << number_of_iters_in_millions << "\t"
					<< average_speed_iters_per_ms << "\t";
			stream << vm_usage_local << "\t" << resident_set_local;
			if (settings.capture_infiniband_stats
					|| settings.capture_ethernet_stats)
				stream << "\t" << received_bytes_local << "\t"
						<< transmitted_bytes_local << "\t"
						<< traffic_mbytes_local;
			if (world != NULL) {
				stream << "\t" << vm_usage_total << "\t" << resident_set_total;
				if (settings.capture_infiniband_stats
						|| settings.capture_ethernet_stats)
					stream << "\t" << received_bytes_total << "\t"
							<< transmitted_bytes_total << "\t"
							<< traffic_mbytes_total;
			}

			break;

		case Verbose:

			stream << " time: " << 1.0e-9 * time_nonstop.elapsed().wall
					<< "s\n CPU time burned across all nodes: "
					<< 1.0e-9 * time_user_total << "s\n objective: "
					<< last_obj_value;
			if (settings.show_prediction_accuracy)
				stream << "\n prediction accuracy: " << prediction_accuracy;
			stream << "\n iterations: " << number_of_iters_in_millions
					<< "M\n speed: " << average_speed_iters_per_ms
					<< "its-per-ms";

			if (world != NULL) {
				stream << "\n total VM usage: " << vm_usage_total
						<< "B\n total RSS: " << resident_set_total << "B";
				if (settings.capture_infiniband_stats
						|| settings.capture_ethernet_stats)
					stream << "\n total traffic: " << traffic_mbytes_total
							<< "MB";
			}

			stream << std::endl;
			break;
		}

		return stream;

	}

	std::ostream& print_dimensions(std::ostream& stream, LogFormat format =
			RawData) {

		if (world != NULL and world->rank() != 0)
			return stream;

		switch (format) {

		case RawHeaders:

			stream << "instance" << "\t" << "columns" << "\t" << "rows" << "\t"
					<< "max_nnz_per_col" << "\t" << "max_nnz_per_row" << "\t"
					<< "parts" << "\t" << "partitioning" << "\t" << "solver"
					<< "\t" << "hypergraph_cut" << "\t" << "imbalance_columns"
					<< "\t" << "imbalance_nnz" << "\t" << "max_parts/row"
					<< "\t" << "traffic_per_comm_iter [floats]" << "\t"
					<< "rows_with_reduce";
			break;

		case RawData:

			stream << instance_name << "\t" << instance_columns << "\t"
					<< instance_rows << "\t" << max_nnz_per_col << "\t"
					<< max_nnz_per_row << "\t" << parts << "\t"
					<< settings.partitioning << "\t" << settings.distributed
					<< "\t" << hypergraph_cut << "\t" << imbalance_columns
					<< "\t" << imbalance_nnz << "\t" << max_parts_per_row
					<< "\t" << floats_exchanged_per_comm_iter << "\t"
					<< residual_rows_with_reduce;
			break;

		case Verbose:

			stream << " Filename = " << instance_name << ", " << instance_rows
					<< "x" << instance_columns;
			stream << " partitioned into " << parts << " parts by "
					<< settings.partitioning << std::endl;
			stream << " hypergraph cut = " << hypergraph_cut << std::endl;
			stream << " max nnz per row = " << max_nnz_per_row << std::endl;
			stream << " parts per row = " << max_parts_per_row << std::endl;
			stream << " max nnz per col = " << max_nnz_per_col << std::endl;
			stream << " nnz to exchange = " << floats_exchanged_per_comm_iter
					<< std::endl;
			stream << " rows with reduce = " << residual_rows_with_reduce
					<< std::endl << std::endl;
			stream << " Solver is set to " << settings.distributed << std::endl;
			break;

		}

		return stream;
	}

	void next_round() {
		time_rounds += 1;
		time_iterations += settings.iterationsPerThread
				* settings.totalThreads; // * world->size();

		number_of_iters_in_millions = 1.0e-6 * time_iterations;
		average_speed_iters_per_ms = 1.0e6 * time_iterations
				/ time_pure.elapsed().user; // miliseconds from nanoseconds
	}

	void capture_usage(bool initial = false) {
		// MEMORY ALLOCATION
		// based on http://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-run-time-in-c
		using std::string;

		std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);
		// dummy vars for leading entries in stat that we don't care about
		string pid, comm, state, ppid, pgrp, session, tty_nr;
		string tpgid, flags, minflt, cminflt, majflt, cmajflt;
		string user_local, kernel_local, cutime, cstime, priority, nice;
		string zero, itrealvalue, starttime;

		// [jmarecek@frontend03 ~]$ cat /proc/self/stat
		// 29320 (cat) R 28661 29320 28661 34820 29320 4194304 174 0 0 0 0 0 0 0 17 0 1 0 2154007457 60350464 121 18446744073709551615 4194304 4212684 140736315411792 18446744073709551615 256117598032 0 0 0 0 0 0 0 17 6 0 0 0
		// 29320 (cat) R ppid 28661 pgrp 29320 session 28661 tty 34820 flags 29320 minflt 4194304 cmin 174 majflt 0 cmajflt 0 time_local 0 time_local 0 cutime 0 cstime 0 0 17 0 1 0 2154007457 60350464 121 18446744073709551615 4194304 4212684 140736315411792 18446744073709551615 256117598032 0 0 0 0 0 0 0 17 6 0 0 0

		stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
				>> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
				>> user_local >> kernel_local >> cutime >> cstime >> priority
				>> nice >> zero >> itrealvalue >> starttime >> vm_usage_local
				>> resident_set_local; // don't care about the rest

		stat_stream.close();

		long page_size_kb = sysconf(_SC_PAGE_SIZE); // in case x86-64 is configured to use 2MB pages
		resident_set_local *= page_size_kb;

		if (world != NULL) {
			reduce(*world, vm_usage_local, vm_usage_total, std::plus<long>(),
					0);
			reduce(*world, resident_set_local, resident_set_total,
					std::plus<long>(), 0);

			// 	boost::timer::nanosecond_type time_pure_total;
			// boost::timer::nanosecond_type time_user_total;

			reduce(*world, time_pure.elapsed().user, time_pure_total,
					std::plus<boost::timer::nanosecond_type>(), 0);
			reduce(*world, time_nonstop.elapsed().user, time_user_total,
					std::plus<boost::timer::nanosecond_type>(), 0);
		}

		// NETWORK TRAFFIC

		unsigned long long bytes_rx = 0;
		unsigned long long bytes_tx = 0;

		/*
		 if (settings.capture_ethernet_stats) {

		 const int interface_id = 2; // lo, usb, eth0
		 std::ifstream stat_stream("/proc/net/dev", std::ios_base::in);

		 if (settings.verbose && (world != NULL) && (world->rank() == 0)) {
		 std::cout << std::endl;
		 std::ifstream another("/proc/net/dev", ios_base::in);
		 std::copy(std::istream_iterator<std::string>(another), std::istream_iterator<std::string>(),
		 std::ostream_iterator<std::string>(std::cout, " "));
		 another.close();
		 std::cout << std::endl;
		 }

		 //			 Inter-|   Receive                                                |  Transmit
		 //			 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
		 //			 lo:1600819596131 2180649632    0    0    0     0          0         0 1600819596131 2180649632    0    0    0     0       0          0
		 //			 eth0:102812914838658 80717483968    0 27733    0     0          0       983 32759667268174 40307527194    0    0    0     0       0          0

		 //			 1 ter-|   2eceive                                                3  4ransmit
		 //			 5ace 6bytes    7ackets 8rrs 9rop 10fo 11ame 12mpressed 13lticast|bytes    14ckets 15rs 16op 17fo 18lls 19rrier 12mpressed

		 std::string dummy;
		 int i;
		 for (i = 0; stat_stream.good() && i < 20; i++)
		 stat_stream >> dummy; // header;

		 unsigned long long packets_rx, errs_rx, drop_rx, fifo_rx, frame_rx, compressed_rx, multicast_rx,
		 packets_tx, errs_tx, drop_tx, fifo_tx, frame_tx, compressed_tx, multicast_tx;

		 for (i = 0; stat_stream.good() && i <= interface_id; i++) {
		 stat_stream >> bytes_rx >> packets_rx >> errs_rx >> drop_rx >> fifo_rx >> frame_rx >> compressed_rx
		 >> multicast_rx;
		 stat_stream >> bytes_tx >> packets_tx >> errs_tx >> drop_tx >> fifo_tx >> frame_tx >> compressed_tx
		 >> multicast_tx;
		 }

		 if (i != interface_id) { // sanity
		 settings.capture_ethernet_stats = false;
		 }

		 stat_stream.close();
		 }

		 if (settings.capture_infiniband_stats) {

		 std::ifstream ib1("/sys/class/infiniband/qib0/ports/1/counters/port_rcv_data", ios_base::in);
		 if (ib1.good())
		 ib1 >> bytes_rx;
		 else
		 settings.capture_infiniband_stats = false;
		 ib1.close();

		 std::ifstream ib2("/sys/class/infiniband/qib0/ports/1/counters/port_xmit_data", ios_base::in);
		 if (ib2.good())
		 ib2 >> bytes_tx;
		 else
		 settings.capture_infiniband_stats = false;
		 ib2.close();

		 }

		 if (settings.capture_infiniband_stats || settings.capture_ethernet_stats) {

		 if (initial) {
		 received_bytes_initial = bytes_rx;
		 transmitted_bytes_initial = bytes_tx;
		 }

		 received_bytes_local = bytes_rx - received_bytes_initial;
		 transmitted_bytes_local = bytes_tx - transmitted_bytes_initial;
		 traffic_mbytes_local = (1.0 * received_bytes_local + transmitted_bytes_local) / (1024.0 * 1024);

		 if (world != NULL) {
		 reduce(*world, received_bytes_local, received_bytes_total, std::plus<long long>(), 0);
		 reduce(*world, transmitted_bytes_local, transmitted_bytes_total, std::plus<long long>(), 0);
		 }

		 traffic_mbytes_total = (1.0 * received_bytes_total + transmitted_bytes_total) / (1024.0 * 1024);
		 }
		 */
	}

};

std::ostream& operator<<(std::ostream& stream, distributed_statistics& stat) {
	stat.print_results(stream, Verbose);
	return stream;
}

void print_line_about_times(char* label, boost::timer::cpu_timer &timer) {
	cout << label << " " << boost::timer::format(timer.elapsed(), 6, "%w") << //wall time
			" " << boost::timer::format(timer.elapsed(), 6, "%u") << //user time
			" " << boost::timer::format(timer.elapsed(), 6, "%s") << //system time
			" " << boost::timer::format(timer.elapsed(), 6, "%t") << // user + system
			" " << boost::timer::format(timer.elapsed(), 6, "%p") << endl; // user+system  / wall
}

void get_additional_times(distributed_statistics &stat) {
//FIXME un-comment
	//	print_line_about_times("Total computation", stat.time_nonstop);
//	print_line_about_times("Total computation", stat.time_all_computation);
//	print_line_about_times("Looses updates   ", stat.time_pure);
//	print_line_about_times("Initialization   ", stat.time_initialization);
//	print_line_about_times("Residuals Shift  ", stat.time_residual_exchange);
//	print_line_about_times("Sampling coordina", stat.time_sample_coordinates);
//	print_line_about_times("Other            ", stat.time_other_overhed);

}

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

int parseLine(char* line) {
	int i = strlen(line);
	while (*line < '0' || *line > '9')
		line++;
	line[i - 3] = '\0';
	i = atoi(line);
	return i;
}

long getTotalSystemMemory() {
	long pages = sysconf(_SC_PHYS_PAGES);
	long page_size = sysconf(_SC_PAGE_SIZE);
	return pages * page_size;
}

int getVirtualMemoryCurrentlyUsedByCurrentProcess() { //Note: this value is in KB!
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmSize:", 7) == 0) {
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	return result;
}

int getPhysicalMemoryCurrentlyUsedByCurrentProcess() { //Note: this value is in KB!
	FILE* file = fopen("/proc/self/status", "r");
	int result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmRSS:", 6) == 0) {
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	return result;
}

void print_current_memory_consumption(mpi::communicator &world) {

	std::cout << "NOTE: MEM VM " << world.rank() << "  "
			<< getVirtualMemoryCurrentlyUsedByCurrentProcess() << " KB "
			<< getPhysicalMemoryCurrentlyUsedByCurrentProcess() << " KB"
			<< std::endl;
}
#include <iostream>
#include <iomanip>
void logStatisticsToFile(ofstream &myOutputStream, distributed_statistics &stat,
		DistributedSettings &settings, string row_prefix = "", bool justStore =
				false) {
	if (!justStore) {
		std::cout << " (" << settings.distributed << ")" << std::endl
				<< ": Objective " << stat.last_obj_value << " Runtime "
				<< boost::timer::format(stat.time_nonstop.elapsed(), 6, "%w")
				<< "\n" << std::endl;
	}
	myOutputStream << setprecision(16) << row_prefix << settings.distributed
			<< "," << stat.total_mpi_processes << "," << settings.torus_width
			<< "," << settings.iterationsPerThread << ","
			<< settings.iters_communicate_count << "," << setprecision(16)
			<< stat.last_obj_value << ","
			<< boost::timer::format(stat.time_all_computation.elapsed(), 6,
					"%w") << ","
			<< boost::timer::format(stat.time_pure.elapsed(), 6, "%w") << ","
			<< boost::timer::format(stat.time_initialization.elapsed(), 6, "%w")
			<< ","
			<< boost::timer::format(stat.time_residual_exchange.elapsed(), 6,
					"%w") << ","
			<< boost::timer::format(stat.time_residual_exchange.elapsed(), 6,
					"%s") << ","
			<< boost::timer::format(stat.time_sample_coordinates.elapsed(), 6,
					"%w") << ","
			<< boost::timer::format(stat.time_other_overhed.elapsed(), 6, "%w")
			<< "," << stat.m_total << "," << stat.prediction_accuracy << ","
			<< stat.sigma << std::endl;
}

#endif /* DISTRIBUTED_STRUCTURES_H_ */

