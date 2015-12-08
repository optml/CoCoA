#ifndef ACDC_DISTRIBUTED_SOLVER
#define ACDC_DISTRIBUTED_SOLVER

#include "../../helpers/gsl_random_helper.h"

#include "distributed_include.h"

#include "distributed_structures.h"
#include "distributed_common.h"
#include "data_distributor.h"
#include "distributed_synchronous.h"
#include "distributed_synchronous_schema.h"
#include "distributed_asynchronous.h"
#include "distributed_svm.h"

#ifdef PROFILING
#define NPROF(message) {}
#define PROF(message) \
    if(world.rank()==0){stat.time_pure.stop();std::cout<<__LINE__<<" "<<std::setw(50)<<message<<": "\
      <<boost::timer::format(stat.time_pure.elapsed(),10,"%w")<<" "\
      <<boost::timer::format(stat.time_nonstop.elapsed(),6,"%w")<<" "\
      <<std::endl;stat.time_pure.start();}
#else
#define NPROF(message) {}
#define PROF(message) {}
#endif

// function object
template<typename D, typename L>
struct GreaterNoNnz: public std::binary_function<L, L, bool> {
	GreaterNoNnz(std::vector<L> &whereToCompare) :
			a(whereToCompare) {
	}
	inline bool operator()(const L& i, const L& j) {
		const L one = a[i + 1] - a[i];
		const L second = a[j + 1] - a[j];
		if (one == second)
			return i < j;
		else
			return one > second; //FIXME find out if this is correct or should be "<" instead
	}
protected:
	std::vector<L> &a;
};

template<typename D, typename L>
inline void exchange_residuals(mpi::environment &env, mpi::communicator &world,
		std::vector<D> &residuals, std::vector<D> &residual_updates,
		ProblemData<L, D> &part, data_distributor<L, D> &dataDistributor,
		distributed_statistics &stat, std::list<std::vector<D> > &buffer,
		std::vector<D> &past_update, DistributedSettings &settings,
		mpi::communicator &rung, std::vector<D> &exchanged, int &exchange_data,
		int &AsynchronousStreamlinedOptimizedUpdate,
		std::vector<D> &residual_updates_tosum) {

	switch (settings.distributed) {
	case SynchronousReduce:
		reduce_residuals(world, residuals, residual_updates, exchanged);
		break;
	case SynchronousGather:
		gather_residuals(world, residuals, residual_updates, part);
		break;
	case SynchronousPointToPoint:
	case SynchronousSparse:
		// NOTE: shift_residuals_point_to_point now extends shift_residuals_sparse
		shift_residuals_point_to_point(env, world, dataDistributor, residuals,
				residual_updates, stat.time_rounds);
		break;

	case SynchronousSupersparse:
		// FIXME: Supersparse should be fixed
		shift_residuals_supersparse(env, world, dataDistributor, residuals,
				residual_updates, settings, stat.time_rounds);
		break;

	case AsynchronousStreamlined:
		// NOTE: Token ring is a torus of width 1, so we could use that
		shift_residuals_buffered_streamlined(env, world, buffer, past_update,
				residuals, residual_updates, settings, stat.time_rounds);
		break;

	case AsynchronousStreamlinedV2:
		// NOTE: Token ring is a torus of width 1, so we could use that
		shift_residuals_buffered_streamlined_v2(env, world, buffer, past_update,
				residuals, residual_updates, settings, stat.time_rounds);
		break;

	case AsynchronousStreamlinedOptimized:
		// NOTE: Token ring is a torus of width 1, so we could use that
		shift_residuals_buffered_streamlined_optimized(env, world,
				residual_updates, residuals, past_update, exchanged,
				exchange_data, AsynchronousStreamlinedOptimizedUpdate, settings,
				stat.time_rounds);
		break;

	case AsynchronousTorus:
		shift_residuals_torus(env, world, residual_updates_tosum, exchanged,
				buffer, past_update, residuals, residual_updates, settings,
				stat.time_rounds);
		break;

	case AsynchronousTorusOpt: {
		shift_residuals_torus_opt(env, world, residual_updates_tosum, exchanged,
				buffer, past_update, residuals, residual_updates, settings,
				stat.time_rounds, settings.topology.local_rung_communicator);
	}
		break;
	case AsynchronousTorusOptCollectives: {
		shift_residuals_torus_opt_collectives(env, world,
				residual_updates_tosum, exchanged, buffer, past_update,
				residuals, residual_updates, settings, stat.time_rounds,
				settings.topology.local_rung_communicator);
	}
		break;

	case AsynchronousTorusCollectives:
		shift_residuals_torus_collectives(env, world, rung, buffer, past_update,
				residuals, residual_updates, settings, stat.time_rounds);
		break;

	}
}

template<typename D, typename L, typename LT>
void bulkIterations_for_distributed_multisource_solver(
		mpi::environment &env, mpi::communicator &world,
		ProblemData<L, D> &part, ProblemData<L, D> &inst,
		data_distributor<L, D> &dataDistributor,
		std::vector<std::vector<D> > &all_x,
		std::vector<D> &buffer_residuals_inst,
		std::vector<std::vector<D> > &buffer_all_residuals,
		std::vector<D> &residuals) {
	gather(world, part.x, all_x, 0);
	if (world.rank() == 0) {
		std::vector<L> used(world.size(), 0);
		for (L i = 0; i < inst.x.size(); i++) {
			L from_part = dataDistributor.columns_parts[i];
			inst.x[i] = all_x[from_part][used[from_part]];
			used[from_part]++;
		}
	}
	mpi::broadcast(world, inst.x, 0);
	// Now all have correct "x" and can compute own residuals
	DistributedLosses<L, D, LT>::bulkIterations_for_my_instance_data(inst,
			buffer_residuals_inst);
	gather(world, buffer_residuals_inst, buffer_all_residuals, 0);
	if (world.rank() == 0) {
		//aggregate all residuals into instance
		for (int i = 0; i < world.size(); i++) {
			for (int j = 0; j < buffer_all_residuals[i].size(); j++) {
				residuals[j + dataDistributor.global_row_id_mapper[i]] =
						buffer_all_residuals[i][j];
			}
		}
	}
	mpi::broadcast(world, residuals, 0);
}

template<typename D, typename L>
void initialize_buffers(mpi::environment &env, mpi::communicator &world,
		std::list<std::vector<D> > &buffer, std::vector<D> &past_update,
		std::vector<D> &residual_updates_tosum, std::vector<D> &exchanged,
		mpi::communicator &rung, DistributedSettings &settings,
		ProblemData<L, D> &part) {

	if (settings.distributed == SynchronousReduce) {
		exchanged.resize(part.m);
	}

	int buffer_size = world.size() - 1;
	if (settings.distributed == AsynchronousBuffered) {
		std::vector<D> empty_v(part.m, 0);
		buffer.resize(buffer_size, empty_v);
	}

	if (settings.distributed == AsynchronousStreamlined) {
		past_update.resize(part.m, 0);
		buffer.resize(buffer_size, past_update);
	}

	if (settings.distributed == AsynchronousStreamlinedV2) {
		past_update.resize(part.m, 0);
		buffer.resize(buffer_size, past_update);
	}

	if (settings.distributed == AsynchronousTorus) {
		past_update.resize(part.m, 0);
		buffer_size = world.size() / settings.torus_width;
		buffer.resize(buffer_size, past_update);
		residual_updates_tosum.resize(part.m * (settings.torus_width - 1), 0);
		exchanged.resize(part.m, 0);
	}

	if (settings.distributed == AsynchronousTorusOpt
			|| settings.distributed == AsynchronousTorusOptCollectives) {

		int rung_root = settings.topology.this_rung_index(world.size(),
				world.rank(), settings.torus_width, 0);
		if (rung_root == world.rank()) {
			settings.topology.local_rung_communicator = world.split(rung_root,
					0);
		} else {
			settings.topology.local_rung_communicator = world.split(rung_root);
		}
		past_update.resize(part.m, 0);
		buffer_size = world.size() / settings.torus_width;
		buffer.resize(buffer_size, past_update);
		residual_updates_tosum.resize(part.m * (settings.torus_width - 1), 0);
		exchanged.resize(part.m, 0);
	}

	if (settings.distributed == AsynchronousTorusCollectives) {
		past_update.resize(part.m, 0);
		buffer_size = world.size() / settings.torus_width;
		buffer.resize(buffer_size, past_update);
		torus_collectives_prepare(env, world, rung, settings);
	}

	if (settings.distributed == AsynchronousStreamlinedOptimized) {
		past_update.resize(part.m * (world.size()), 0);
		exchanged.resize(part.m * 2, 0);
	}
}

template<typename D, typename L, typename LT>
void distributed_solver_from_multiple_sources(mpi::environment &env,
		mpi::communicator &world, DistributedSettings &settings,
		distributed_statistics &stat, ProblemData<L, D> &part,
		data_distributor<L, D> &dataDistributor, std::vector<gsl_rng *> &rs) {

	omp_set_num_threads(settings.totalThreads);
	randomNumberUtil::init_random_seeds(rs,
			settings.totalThreads * world.rank() * world.size());

	world.barrier();
	stat.time_initialization.start();
	stat.time_all_computation.start();
	// Solver-specific structures
#ifdef NEWATOMICS
	std::vector< atomic_float<D, L> > residuals(part.m); // compute residuals = -b;
#else
	std::vector<D> residuals(part.m); // compute residuals = -b;
#endif
	std::vector<D> buffer_residuals(part.m);

	/*
	 * Prepare Residuals and Strucuters which helps to exchange residuals
	 */
	Losses<L, D, LT>::set_residuals_for_zero_x(part, residuals);
	// Method-specific structures
	std::list < std::vector<D> > buffer;
	std::vector<D> past_update;
	std::vector<D> residual_updates_tosum;
	std::vector<D> exchanged;

	mpi::communicator rung;
	int exchange_data = 0;
	int AsynchronousStreamlinedOptimizedUpdate = 0;
	initialize_buffers(env, world, buffer, past_update, residual_updates_tosum,
			exchanged, rung, settings, part);
	// Clear local "x"
	part.x.resize(part.n, 0);
	cblas_set_to_zero(part.x);

	L round = 0; // the round of communication
	// Setting up the parallelisation
//	omp_set_num_threads(settings.totalThreads); //FIXME

	/* TODO FIX!!!!!!!!!!!!!!*/

	unsigned int columns_per_inner = settings.iterationsPerThread
			* settings.totalThreads;
	std::vector<L> inner_idx(columns_per_inner);
	unsigned int inner_seed[columns_per_inner];
	srand(world.rank());
	for (int i = 0; i < columns_per_inner; i++) {
		inner_seed[i] = rand();
	}
	/*=================*/

	std::vector<D> h_Li(part.n, 0);
	Losses<L, D, LT>::compute_reciprocal_lipschitz_constants(part, h_Li);

	cout << "part m = " << part.m << endl;

	std::vector<D> residual_updates;
	D* resudial_updates_ptr;
	if (settings.distributed != AsynchronousStreamlinedOptimized) {
		residual_updates.resize(part.m, 0);
		resudial_updates_ptr = &residual_updates[0];
	}
	randomNumberUtil::init_omp_random_seeds(world.rank());
	stat.time_initialization.stop();

	settings.showIntermediateObjectiveValue = true;
	if (settings.showIntermediateObjectiveValue) {
		stat.last_obj_value =
				DistributedLosses<L, D, LT>::compute_fast_objective(part,
						residuals, world);
		if (world.rank() == 0) {
			logStatisticsToFile(myfile, stat, settings, "detail_", true);
		}
	}

	stat.time_residual_exchange.start();
	stat.time_residual_exchange.stop();
	stat.time_pure.start();
	stat.time_pure.stop();
	stat.time_sample_coordinates.start();
	stat.time_sample_coordinates.stop();
	stat.time_other_overhed.start();
	stat.time_other_overhed.stop();

	for (L rr_iteration = 0;
			rr_iteration < settings.iters_bulkIterations_count;
			rr_iteration++) {
		PROF("RR iteration")
		stat.time_pure.resume();
		for (L comm_iteration = 0;
				comm_iteration < settings.iters_communicate_count;
				comm_iteration++) {

			if (settings.distributed == AsynchronousStreamlinedOptimized) {
				resudial_updates_ptr =
						&past_update[AsynchronousStreamlinedOptimizedUpdate
								* part.m];
			}

			cblas_set_to_zero(part.m, resudial_updates_ptr);
			// TODO: Do this in an extra thread, in advance
			PROF("Sorting indices")
			stat.time_pure.stop();
			stat.time_sample_coordinates.resume();

//			for (int i = 0; i < columns_per_inner; i++) {
//				L idx = (L) (part.n * ((float) rand_r(&inner_seed[i]) / RAND_MAX));
//				if (idx == part.n)
//					idx--;
//				inner_idx[i] = idx;
//			}
//			GreaterNoNnz<D, L> mycomp(part.A_csc_col_ptr);
//			std::sort(inner_idx.begin(), inner_idx.end(), mycomp);
			stat.time_sample_coordinates.stop();
			stat.time_pure.resume();
			// PROF("OpenMP for loop")
#pragma omp parallel
			{
#pragma omp for
				for (int i = 0; i < columns_per_inner; i++) {
//					L idx = (L) (part.n * ((float) rand_r(&myseed) / RAND_MAX));
//					if (idx == part.n)
//						idx--;
					L idx = gsl_rng_uniform_int(gsl_rng_r, part.n);

					Losses<L, D, LT>::do_single_iteration_parallel_for_distributed(
							part, idx, residuals, part.x, h_Li,
							resudial_updates_ptr);
				}
			}
			stat.time_pure.stop();
			PROF("residuals shift")
			stat.time_residual_exchange.resume();

			exchange_residuals(env, world, residuals, residual_updates, part,
					dataDistributor, stat, buffer, past_update, settings, rung,
					exchanged, exchange_data,
					AsynchronousStreamlinedOptimizedUpdate,
					residual_updates_tosum);

			stat.time_residual_exchange.stop();
			PROF("next_round")
			stat.next_round();
		}
		// FIXME REMOVE
		//		if (settings.distributed == SynchronousReduce || settings.distributed == SynchronousGather
		//				|| settings.distributed == SynchronousPointToPoint || settings.distributed == SynchronousSparse
		//				|| settings.distributed == SynchronousSupersparse) {
		//			PROF("barrier wait")
		//			world.barrier();
		//		}

		// NOTE: This should NOT be run only at world.rank() == 0, because we want to reduce!
		stat.capture_usage();
		stat.time_other_overhed.resume();
		if (settings.bulkIterations) {
			PROF("recompute residuals")
			DistributedLosses<L, D, LT>::bulkIterations_for_my_part_data(
					part, residuals, buffer_residuals, world);
			// Residuals have been synchronized, one needs to clean buffers
			if (settings.distributed == AsynchronousStreamlined
					|| settings.distributed == AsynchronousStreamlinedV2
					|| settings.distributed == AsynchronousTorus
					|| settings.distributed == AsynchronousTorusOpt
					|| settings.distributed == AsynchronousTorusOptCollectives
					|| settings.distributed == AsynchronousTorusCollectives) {
				PROF("buffer cleaning")
				cblas_set_to_zero(past_update);
				for (typename std::list<std::vector<D> >::iterator list_iter =
						buffer.begin(); list_iter != buffer.end();
						list_iter++) {
					cblas_set_to_zero((*list_iter));
				}
			} else if (settings.distributed
					== AsynchronousStreamlinedOptimized) {
				cblas_set_to_zero(exchanged);
				cblas_set_to_zero(past_update);
			}
		}

		if (settings.showIntermediateObjectiveValue) {
			stat.last_obj_value =
					DistributedLosses<L, D, LT>::compute_fast_objective(part,
							residuals, world);
			if (world.rank() == 0) {
				//stat.print_results_to_log();
				cout << stat.last_obj_value << ","
						<< stat.generated_optimal_value << ","
						<< stat.last_obj_value - stat.generated_optimal_value
						<< "," << rr_iteration << ",IVAL" << endl;
//				printf("LAST Objective %f   %f\n", stat.last_obj_value, part.lambda);
			}
		}
		if (settings.show_prediction_accuracy) {
			PROF("prediction accuracy")
			// NOTE: This should not be run only at world.rank() == 0, because we want to reduce!
			//FIXME Implement!!!
			/*
			 stat.prediction_accuracy = compute_distributed_prediction_accuracy_for_svm(env, world, inst, inst);
			 */
			stat.prediction_accuracy =
					compute_distributed_prediction_accuracy_for_svm_for_part(
							env, world, part);
		}

		if (settings.showIntermediateObjectiveValue) {
			logStatisticsToFile(myfile, stat, settings, "detail_", true);
		}

		stat.time_other_overhed.stop();

	}
	stat.time_all_computation.stop();
}

template<typename D, typename L, typename LT>
void distributed_solver_from_multiple_sources_structured(mpi::environment &env,
		mpi::communicator &world, DistributedSettings &settings,
		distributed_statistics &stat, ProblemData<L, D> &part_local,
		ProblemData<L, D> &part_global,
		data_distributor<L, D> &dataDistributor, std::vector<gsl_rng *> &rs) {

	omp_set_num_threads(settings.totalThreads);
	randomNumberUtil::init_random_seeds(rs,
			settings.totalThreads * world.rank() * world.size());

	world.barrier();
	stat.time_initialization.start();
	stat.time_all_computation.start();
	// Solver-specific structures
	std::vector<D> residuals_global(part_global.m); // compute residuals_global = -b;
	std::vector<D> buffer_residuals(part_global.m);

	std::vector<D> residuals_local(part_local.m); // compute residuals_global = -b;

	/*
	 * Prepare Residuals and Strucuters which helps to exchange residuals_global
	 */
	Losses<L, D, LT>::set_residuals_for_zero_x(part_global, residuals_global);
	Losses<L, D, LT>::set_residuals_for_zero_x(part_local, residuals_local);
	// Method-specific structures
	std::list < std::vector<D> > buffer;
	std::vector<D> past_update;
	std::vector<D> residual_updates_tosum;
	std::vector<D> exchanged;

	mpi::communicator rung;
	int exchange_data = 0;
	int AsynchronousStreamlinedOptimizedUpdate = 0;
	initialize_buffers(env, world, buffer, past_update, residual_updates_tosum,
			exchanged, rung, settings, part_global);
	// Clear local "x"
	part_global.x.resize(part_global.n, 0);
	cblas_set_to_zero(part_global.x);

	L round = 0; // the round of communication
	// Setting up the parallelisation
//	omp_set_num_threads(settings.totalThreads); //FIXME

	/* TODO FIX!!!!!!!!!!!!!!*/

	unsigned int columns_per_inner = settings.iterationsPerThread
			* settings.totalThreads;
	std::vector<L> inner_idx(columns_per_inner);
	unsigned int inner_seed[columns_per_inner];
	srand(world.rank());
	for (int i = 0; i < columns_per_inner; i++) {
		inner_seed[i] = rand();
	}
	/*=================*/

	std::vector<D> h_Li(part_global.n, 0);
	Losses<L, D, LT>::compute_reciprocal_lipschitz_constants(part_global,
			part_local, h_Li);

	cout << "part m = " << part_global.m << endl;

	std::vector<D> residual_updates;
	D* resudial_updates_ptr;
	if (settings.distributed != AsynchronousStreamlinedOptimized) {
		residual_updates.resize(part_global.m, 0);
		resudial_updates_ptr = &residual_updates[0];
	}
	randomNumberUtil::init_omp_random_seeds(world.rank());
	stat.time_initialization.stop();

	settings.showIntermediateObjectiveValue = true;
	if (settings.showIntermediateObjectiveValue) {
		stat.last_obj_value =
				DistributedLosses<L, D, LT>::compute_fast_objective(part_global,
						residuals_global, world);
		if (world.rank() == 0) {
			logStatisticsToFile(myfile, stat, settings, "detail_", true);
		}
	}

	stat.time_residual_exchange.start();
	stat.time_residual_exchange.stop();
	stat.time_pure.start();
	stat.time_pure.stop();
	stat.time_sample_coordinates.start();
	stat.time_sample_coordinates.stop();
	stat.time_other_overhed.start();
	stat.time_other_overhed.stop();

	stat.time_nonstop.start();
	stat.time_nonstop.stop();

	for (L rr_iteration = 0;
			rr_iteration < settings.iters_bulkIterations_count;
			rr_iteration++) {
		stat.time_nonstop.resume();
		for (L comm_iteration = 0;
				comm_iteration < settings.iters_communicate_count;
				comm_iteration++) {
			stat.time_pure.resume();
			if (settings.distributed == AsynchronousStreamlinedOptimized) {
				resudial_updates_ptr =
						&past_update[AsynchronousStreamlinedOptimizedUpdate
								* part_global.m];
			}
			cblas_set_to_zero(part_global.m, resudial_updates_ptr);
#pragma omp parallel
			{
				for (int i = 0; i < columns_per_inner; i++) {
					L idx = gsl_rng_uniform_int(gsl_rng_r, part_global.n);
					Losses<L, D, LT>::do_single_iteration_parallel_for_distributed(
							part_global, part_local, idx, residuals_global,
							residuals_local, part_global.x, h_Li,
							resudial_updates_ptr);
				}
			}
			stat.time_pure.stop();
			PROF("residuals_global shift")
			stat.time_residual_exchange.resume();

			exchange_residuals(env, world, residuals_global, residual_updates,
					part_global, dataDistributor, stat, buffer, past_update,
					settings, rung, exchanged, exchange_data,
					AsynchronousStreamlinedOptimizedUpdate,
					residual_updates_tosum);

			stat.time_residual_exchange.stop();
			PROF("next_round")
			stat.next_round();
		}
		stat.time_nonstop.stop();

		// FIXME REMOVE
		//		if (settings.distributed == SynchronousReduce || settings.distributed == SynchronousGather
		//				|| settings.distributed == SynchronousPointToPoint || settings.distributed == SynchronousSparse
		//				|| settings.distributed == SynchronousSupersparse) {
		//			PROF("barrier wait")
		//			world.barrier();
		//		}

		// NOTE: This should NOT be run only at world.rank() == 0, because we want to reduce!
		stat.capture_usage();
		stat.time_other_overhed.resume();
		if (settings.bulkIterations) {
			PROF("recompute residuals_global")
			DistributedLosses<L, D, LT>::bulkIterations_for_my_part_data(
					part_global, residuals_global, buffer_residuals, world);
			Losses<L, D, LT>::bulkIterations(part_local, residuals_local,
					part_global.x);
			// Residuals have been synchronized, one needs to clean buffers
			if (settings.distributed == AsynchronousStreamlined
					|| settings.distributed == AsynchronousStreamlinedV2
					|| settings.distributed == AsynchronousTorus
					|| settings.distributed == AsynchronousTorusOpt
					|| settings.distributed == AsynchronousTorusOptCollectives
					|| settings.distributed == AsynchronousTorusCollectives) {
				PROF("buffer cleaning")
				cblas_set_to_zero(past_update);
				for (typename std::list<std::vector<D> >::iterator list_iter =
						buffer.begin(); list_iter != buffer.end();
						list_iter++) {
					cblas_set_to_zero((*list_iter));
				}
			} else if (settings.distributed
					== AsynchronousStreamlinedOptimized) {
				cblas_set_to_zero(exchanged);
				cblas_set_to_zero(past_update);
			}
		}

		if (settings.showIntermediateObjectiveValue) {
			stat.last_obj_value =
					DistributedLosses<L, D, LT>::compute_fast_objective(
							part_global, part_local, residuals_global,
							residuals_local, world);
			if (world.rank() == 0) {
				//stat.print_results_to_log();
				cout << stat.last_obj_value << ","
						<< stat.generated_optimal_value << ","
						<< stat.last_obj_value - stat.generated_optimal_value
						<< "," << rr_iteration << ",IVAL" << endl;
//				printf("LAST Objective %f   %f\n", stat.last_obj_value, part.lambda);
			}
		}
		if (settings.show_prediction_accuracy) {
			PROF("prediction accuracy")
			// NOTE: This should not be run only at world.rank() == 0, because we want to reduce!
			//FIXME Implement!!!
			/*
			 stat.prediction_accuracy = compute_distributed_prediction_accuracy_for_svm(env, world, inst, inst);
			 */
			stat.prediction_accuracy =
					compute_distributed_prediction_accuracy_for_svm_for_part(
							env, world, part_global);
		}

		if (settings.showIntermediateObjectiveValue) {
			logStatisticsToFile(myfile, stat, settings, "detail_", true);
		}

		stat.time_other_overhed.stop();

	}
	stat.time_all_computation.stop();
}

#include <boost/thread.hpp>

template<typename D, typename L, typename LT>
void distributed_solver_from_multiple_sources_structured_hybrid(
		mpi::environment &env, mpi::communicator &world,
		DistributedSettings &settings, distributed_statistics &stat,
		ProblemData<L, D> &part_local, ProblemData<L, D> &part_global,
		data_distributor<L, D> &dataDistributor, std::vector<gsl_rng *> &rs) {

	omp_set_num_threads(settings.totalThreads);
	randomNumberUtil::init_random_seeds(rs,
			settings.totalThreads * world.rank() * world.size());

	world.barrier();
	stat.time_initialization.start();
	stat.time_all_computation.start();
	// Solver-specific structures

	std::vector<D> residuals_global(part_global.m); // compute residuals_global = -b;
	std::vector<D> buffer_residuals(part_global.m);

	std::vector<D> residuals_local(part_local.m); // compute residuals_global = -b;

	/*
	 * Prepare Residuals and Strucuters which helps to exchange residuals_global
	 */
	Losses<L, D, LT>::set_residuals_for_zero_x(part_global, residuals_global);
	Losses<L, D, LT>::set_residuals_for_zero_x(part_local, residuals_local);
	// Method-specific structures
	std::list < std::vector<D> > buffer;
	std::vector<D> past_update;
	std::vector<D> residual_updates_tosum;
	std::vector<D> exchanged;

	mpi::communicator rung;
	int exchange_data = 0;
	int AsynchronousStreamlinedOptimizedUpdate = 0;
	initialize_buffers(env, world, buffer, past_update, residual_updates_tosum,
			exchanged, rung, settings, part_global);
	// Clear local "x"
	part_global.x.resize(part_global.n, 0);
	cblas_set_to_zero(part_global.x);

	L round = 0; // the round of communication
	// Setting up the parallelisation
//	omp_set_num_threads(settings.totalThreads); //FIXME

	/* TODO FIX!!!!!!!!!!!!!!*/

	unsigned int columns_per_inner = settings.iterationsPerThread
			* settings.totalThreads;
	std::vector<L> inner_idx(columns_per_inner);
	unsigned int inner_seed[columns_per_inner];
	srand(world.rank());
	for (int i = 0; i < columns_per_inner; i++) {
		inner_seed[i] = rand();
	}
	/*=================*/

	std::vector<D> h_Li(part_global.n, 0);
	Losses<L, D, LT>::compute_reciprocal_lipschitz_constants(part_global,
			part_local, h_Li);
	cout << "part m = " << part_global.m << endl;

	std::vector<D> residual_updates;

	std::vector<D> residual_updates_for_workers(part_global.m, 0);
	D* residual_updates_for_workers_ptr = &residual_updates_for_workers[0];
	cblas_set_to_zero(residual_updates_for_workers);
	D* resudial_updates_ptr;
	if (settings.distributed != AsynchronousStreamlinedOptimized) {
		residual_updates.resize(part_global.m, 0);
		resudial_updates_ptr = &residual_updates[0];
	}
	randomNumberUtil::init_omp_random_seeds(world.rank());
	stat.time_initialization.stop();

	settings.showIntermediateObjectiveValue = true;
	if (settings.showIntermediateObjectiveValue) {
		stat.last_obj_value =
				DistributedLosses<L, D, LT>::compute_fast_objective(part_global,
						residuals_global, world);
		if (world.rank() == 0) {
			logStatisticsToFile(myfile, stat, settings, "detail_", true);
		}
	}

	stat.time_residual_exchange.start();
	stat.time_residual_exchange.stop();
	stat.time_pure.start();
	stat.time_pure.stop();

	stat.time_nonstop.start();
	stat.time_nonstop.stop();

	stat.time_sample_coordinates.start();
	stat.time_sample_coordinates.stop();
	stat.time_other_overhed.start();
	stat.time_other_overhed.stop();

	int columns_per_inner_final_per_thread = columns_per_inner
			* (TOTAL_THREADS - 1) / TOTAL_THREADS;

	float currentWorkerIteration[TOTAL_THREADS];
	const float floatIncrementOne = 1;
	for (L rr_iteration = 0;
			rr_iteration < settings.iters_bulkIterations_count;
			rr_iteration++) {

		PROF("RR iteration")
		stat.time_pure.resume();
		stat.time_nonstop.resume();

		/* =========================================MAIN LOOP =======================*/
//#pragma omp parallel shared(currentWorkerIteration)
//		{
//			if (my_thread_id == 0) {
//				for (L comm_iteration = 0;
//						comm_iteration < settings.iters_communicate_count;
//						comm_iteration++) {
//#pragma omp barrier
//					if (settings.distributed
//							== AsynchronousStreamlinedOptimized) {
//						resudial_updates_ptr =
//								&past_update[AsynchronousStreamlinedOptimizedUpdate
//										* part_global.m];
//					}
//					for (L i = 0; i < residual_updates_for_workers.size();
//							i++) {
//						D tmpVal = residual_updates_for_workers[i];
//						resudial_updates_ptr[i] = tmpVal;
//						parallel::atomic_add(
//								residual_updates_for_workers_ptr[i], -tmpVal);
//					}
//					exchange_residuals(env, world, residuals_global,
//							residual_updates, part_global, dataDistributor,
//							stat, buffer, past_update, settings, rung,
//							exchanged, exchange_data,
//							AsynchronousStreamlinedOptimizedUpdate,
//							residual_updates_tosum);
//				}
//			} else {
//				for (L comm_iteration = 0;
//						comm_iteration < settings.iters_communicate_count;
//						comm_iteration++) {
//					for (int i = 0; i < columns_per_inner_final_per_thread;
//							i++) {
//						L idx = gsl_rng_uniform_int(gsl_rng_r, part_global.n);
//						Losses<L, D, LT>::do_single_iteration_parallel_for_distributed(
//								part_global, part_local, idx, residuals_global,
//								residuals_local, part_global.x, h_Li,
//								residual_updates_for_workers_ptr);
//					}
//#pragma omp barrier
//				}
//			}
//		}
		/* ===========================*/
		for (int i = 0; i < TOTAL_THREADS; i++) {
			currentWorkerIteration[i] = -1;
		}

#pragma omp parallel shared(currentWorkerIteration)
		{
			if (my_thread_id == 0) {
				for (L comm_iteration = 0;
						comm_iteration < settings.iters_communicate_count;
						comm_iteration++) {
					bool allWorkersFinished = false;
					parallel::atomic_add(currentWorkerIteration[0],
							floatIncrementOne);
					while (!allWorkersFinished) {
						allWorkersFinished = true;
						for (int i = 1; i < TOTAL_THREADS; i++) {
							if (currentWorkerIteration[i] < comm_iteration)
								allWorkersFinished = false;
						}
						if (!allWorkersFinished) {
							L idx = gsl_rng_uniform_int(gsl_rng_r,
									part_global.n);
							Losses<L, D, LT>::do_single_iteration_parallel_for_distributed(
									part_global, part_local, idx,
									residuals_global, residuals_local,
									part_global.x, h_Li,
									residual_updates_for_workers_ptr);
							boost::this_thread::sleep(
									boost::posix_time::milliseconds(10));
						}
					}

					if (settings.distributed
							== AsynchronousStreamlinedOptimized) {
						resudial_updates_ptr =
								&past_update[AsynchronousStreamlinedOptimizedUpdate
										* part_global.m];
					}
					for (L i = 0; i < residual_updates_for_workers.size();
							i++) {
						D tmpVal = residual_updates_for_workers[i];
						resudial_updates_ptr[i] = tmpVal;
						parallel::atomic_add(
								residual_updates_for_workers_ptr[i], -tmpVal);
					}
					stat.time_residual_exchange.resume();
					exchange_residuals(env, world, residuals_global,
							residual_updates, part_global, dataDistributor,
							stat, buffer, past_update, settings, rung,
							exchanged, exchange_data,
							AsynchronousStreamlinedOptimizedUpdate,
							residual_updates_tosum);
					stat.time_residual_exchange.stop();
				}

			} else {
				for (L comm_iteration = 0;
						comm_iteration < settings.iters_communicate_count;
						comm_iteration++) {
					for (int i = 0; i < columns_per_inner_final_per_thread;
							i++) {
						L idx = gsl_rng_uniform_int(gsl_rng_r, part_global.n);
						Losses<L, D, LT>::do_single_iteration_parallel_for_distributed(
								part_global, part_local, idx, residuals_global,
								residuals_local, part_global.x, h_Li,
								residual_updates_for_workers_ptr);
					}
					parallel::atomic_add(currentWorkerIteration[my_thread_id],
							floatIncrementOne);
					while (currentWorkerIteration[0] < comm_iteration
							&& comm_iteration
									!= settings.iters_communicate_count - 1) {
						boost::this_thread::sleep(
								boost::posix_time::milliseconds(5));
					}

				}

			}
		}
		/* =========================================MAIN LOOP - ENDS =======================*/

		stat.time_pure.stop();
		stat.time_nonstop.stop();
//	PROF("Sorting indices")
//
//	stat.time_sample_coordinates.resume();
//
//	stat.time_sample_coordinates.stop();
//	stat.time_pure.resume();
//	stat.time_pure.stop();
//	PROF("residuals_global shift")
//	PROF("next_round")
//	stat.next_round();

// NOTE: This should NOT be run only at world.rank() == 0, because we want to reduce!
		stat.capture_usage();
		stat.time_other_overhed.resume();
		if (settings.bulkIterations) {

			cblas_set_to_zero(residual_updates_for_workers);

			PROF("recompute residuals_global")
			DistributedLosses<L, D, LT>::bulkIterations_for_my_part_data(
					part_global, residuals_global, buffer_residuals, world);
			Losses<L, D, LT>::bulkIterations(part_local, residuals_local,
					part_global.x);
			// Residuals have been synchronized, one needs to clean buffers
			if (settings.distributed == AsynchronousStreamlined
					|| settings.distributed == AsynchronousStreamlinedV2
					|| settings.distributed == AsynchronousTorus
					|| settings.distributed == AsynchronousTorusOpt
					|| settings.distributed == AsynchronousTorusOptCollectives
					|| settings.distributed == AsynchronousTorusCollectives) {
				PROF("buffer cleaning")
				cblas_set_to_zero(past_update);
				for (typename std::list<std::vector<D> >::iterator list_iter =
						buffer.begin(); list_iter != buffer.end();
						list_iter++) {
					cblas_set_to_zero((*list_iter));
				}
			} else if (settings.distributed
					== AsynchronousStreamlinedOptimized) {
				cblas_set_to_zero(exchanged);
				cblas_set_to_zero(past_update);
			}
		}

		if (settings.showIntermediateObjectiveValue) {
			stat.last_obj_value =
					DistributedLosses<L, D, LT>::compute_fast_objective(
							part_global, part_local, residuals_global,
							residuals_local, world);
			if (world.rank() == 0) {
				//stat.print_results_to_log();
				cout << stat.last_obj_value << ","
						<< stat.generated_optimal_value << ","
						<< stat.last_obj_value - stat.generated_optimal_value
						<< "," << rr_iteration << ",IVAL" << endl;
//				printf("LAST Objective %f   %f\n", stat.last_obj_value, part.lambda);
			}
		}
		if (settings.show_prediction_accuracy) {
			PROF("prediction accuracy")
			stat.prediction_accuracy =
					compute_distributed_prediction_accuracy_for_svm_for_part(
							env, world, part_global);
		}
		if (settings.showIntermediateObjectiveValue) {
			logStatisticsToFile(myfile, stat, settings, "detail_", true);
		}
		stat.time_other_overhed.stop();
	}
	stat.time_all_computation.stop();
}

template<typename D, typename L, typename LT>
void distributed_solver_from_multiple_sources_structured_hybrid_barrier(
		mpi::environment &env, mpi::communicator &world,
		DistributedSettings &settings, distributed_statistics &stat,
		ProblemData<L, D> &part_local, ProblemData<L, D> &part_global,
		data_distributor<L, D> &dataDistributor, std::vector<gsl_rng *> &rs) {
	stat.sigma = part_global.sigma;
	omp_set_num_threads(settings.totalThreads);
	randomNumberUtil::init_random_seeds(rs,
			settings.totalThreads * world.rank() * world.size());

	if (world.rank() == 0) {
		cout << "Using " << settings.distributed << " with"
				<< settings.iters_communicate_count << endl;
	}

	world.barrier();
	stat.time_initialization.start();
	stat.time_all_computation.start();
	// Solver-specific structures

	std::vector<D> residuals_global(part_global.m); // compute residuals_global = -b;
	std::vector<D> buffer_residuals(part_global.m);

	std::vector<D> residuals_local(part_local.m); // compute residuals_global = -b;

	/*
	 * Prepare Residuals and Strucuters which helps to exchange residuals_global
	 */
	Losses<L, D, LT>::set_residuals_for_zero_x(part_global, residuals_global);
	Losses<L, D, LT>::set_residuals_for_zero_x(part_local, residuals_local);
	// Method-specific structures
	std::list < std::vector<D> > buffer;
	std::vector<D> past_update;
	std::vector<D> residual_updates_tosum;
	std::vector<D> exchanged;

	mpi::communicator rung;
	int exchange_data = 0;
	int AsynchronousStreamlinedOptimizedUpdate = 0;
	initialize_buffers(env, world, buffer, past_update, residual_updates_tosum,
			exchanged, rung, settings, part_global);
	// Clear local "x"
	part_global.x.resize(part_global.n, 0);
	cblas_set_to_zero(part_global.x);

	L round = 0; // the round of communication
	// Setting up the parallelisation
//	omp_set_num_threads(settings.totalThreads); //FIXME

	/* TODO FIX!!!!!!!!!!!!!!*/

	unsigned int columns_per_inner = settings.iterationsPerThread;
	std::vector<L> inner_idx(columns_per_inner);
	unsigned int inner_seed[columns_per_inner];
	srand(world.rank());
	for (int i = 0; i < columns_per_inner; i++) {
		inner_seed[i] = rand();
	}
	/*=================*/
	std::vector<D> h_Li(part_global.n, 0);
	Losses<L, D, LT>::compute_reciprocal_lipschitz_constants(part_global,
			part_local, h_Li);


	std::vector<D> residual_updates_for_workers(part_global.m, 0);
	D* residual_updates_for_workers_ptr = &residual_updates_for_workers[0];
	cblas_set_to_zero(residual_updates_for_workers);
	D* resudial_updates_ptr;
	std::vector<D> residual_updates;
	if (settings.distributed != AsynchronousStreamlinedOptimized) {
		residual_updates.resize(part_global.m, 0);
		resudial_updates_ptr = &residual_updates[0];
	}
	randomNumberUtil::init_omp_random_seeds(world.rank());
	stat.time_initialization.stop();

	settings.showIntermediateObjectiveValue = true;
	if (settings.showIntermediateObjectiveValue) {

		DistributedLosses<L, D, LT>::bulkIterations_for_my_part_data(
				part_global, residuals_global, buffer_residuals, world);

		stat.last_obj_value =
				DistributedLosses<L, D, LT>::compute_fast_objective(part_global,
						part_local, residuals_global, residuals_local, world);

		if (world.rank() == 0) {
			cout << "IVAL FIRST " << stat.last_obj_value << endl;
			logStatisticsToFile(myfile, stat, settings, "detail_", true);
		}
	}

	stat.time_residual_exchange.start();
	stat.time_residual_exchange.stop();
	stat.time_pure.start();
	stat.time_pure.stop();

	stat.time_nonstop.start();
	stat.time_nonstop.stop();

	stat.time_sample_coordinates.start();
	stat.time_sample_coordinates.stop();
	stat.time_other_overhed.start();
	stat.time_other_overhed.stop();

	int columns_per_inner_final_per_thread = columns_per_inner
			* TOTAL_THREADS / (TOTAL_THREADS - 1.0) ;

	float currentWorkerIteration[TOTAL_THREADS];
	const float floatIncrementOne = 1;

	L rr_iteration = 0;

	int continue_work = 1;

	while (continue_work == 1
			&& rr_iteration < settings.iters_bulkIterations_count) {

		PROF("RR iteration")
		stat.time_pure.resume();
		stat.time_nonstop.resume();

		/* =========================================MAIN LOOP =======================*/
#pragma omp parallel shared(currentWorkerIteration)
		{
			if (my_thread_id == 0) {
				for (L comm_iteration = 0;
						comm_iteration < settings.iters_communicate_count;
						comm_iteration++) {
#pragma omp barrier
					if (settings.distributed
							== AsynchronousStreamlinedOptimized) {
						resudial_updates_ptr =
								&past_update[AsynchronousStreamlinedOptimizedUpdate
										* part_global.m];
					}
					for (L i = 0; i < residual_updates_for_workers.size();
							i++) {
						D tmpVal = residual_updates_for_workers[i];
						resudial_updates_ptr[i] = tmpVal;
						parallel::atomic_add(
								residual_updates_for_workers_ptr[i], -tmpVal);
					}
					exchange_residuals(env, world, residuals_global,
							residual_updates, part_global, dataDistributor,
							stat, buffer, past_update, settings, rung,
							exchanged, exchange_data,
							AsynchronousStreamlinedOptimizedUpdate,
							residual_updates_tosum);
				}
			} else {
				for (L comm_iteration = 0;
						comm_iteration < settings.iters_communicate_count;
						comm_iteration++) {
					for (int i = 0; i < columns_per_inner_final_per_thread;
							i++) {
						L idx = gsl_rng_uniform_int(gsl_rng_r, part_global.n);
						Losses<L, D, LT>::do_single_iteration_parallel_for_distributed(
								part_global, part_local, idx, residuals_global,
								residuals_local, part_global.x, h_Li,
								residual_updates_for_workers_ptr);

					}
#pragma omp barrier
				}
			}
		}

		/* =========================================MAIN LOOP - ENDS =======================*/
		stat.time_nonstop.stop();
		stat.time_pure.stop();

//	PROF("Sorting indices")
//
//	stat.time_sample_coordinates.resume();
//
//	stat.time_sample_coordinates.stop();
//	stat.time_pure.resume();
//	stat.time_pure.stop();
//	PROF("residuals_global shift")
//	PROF("next_round")
//	stat.next_round();

// NOTE: This should NOT be run only at world.rank() == 0, because we want to reduce!
		stat.capture_usage();
		stat.time_other_overhed.resume();
		if (settings.bulkIterations) {

			cblas_set_to_zero(residual_updates_for_workers);

			PROF("recompute residuals_global")
			DistributedLosses<L, D, LT>::bulkIterations_for_my_part_data(
					part_global, residuals_global, buffer_residuals, world);
			Losses<L, D, LT>::bulkIterations(part_local, residuals_local,
					part_global.x);
			// Residuals have been synchronized, one needs to clean buffers
			if (settings.distributed == AsynchronousStreamlined
					|| settings.distributed == AsynchronousStreamlinedV2
					|| settings.distributed == AsynchronousTorus
					|| settings.distributed == AsynchronousTorusOpt
					|| settings.distributed == AsynchronousTorusOptCollectives
					|| settings.distributed == AsynchronousTorusCollectives) {
				PROF("buffer cleaning")
				cblas_set_to_zero(past_update);
				for (typename std::list<std::vector<D> >::iterator list_iter =
						buffer.begin(); list_iter != buffer.end();
						list_iter++) {
					cblas_set_to_zero((*list_iter));
				}
			} else if (settings.distributed
					== AsynchronousStreamlinedOptimized) {
				cblas_set_to_zero(exchanged);
				cblas_set_to_zero(past_update);
			}
		}

		if (settings.showIntermediateObjectiveValue) {
			stat.last_obj_value =
					DistributedLosses<L, D, LT>::compute_fast_objective(
							part_global, part_local, residuals_global,
							residuals_local, world);
			continue_work = 0;
			if (world.rank() == 0) {
				//stat.print_results_to_log();
				cout << stat.last_obj_value << ","
						<< stat.generated_optimal_value << ","
						<< stat.last_obj_value - stat.generated_optimal_value
						<< "," << rr_iteration << ",IVAL" << ","
						<< boost::timer::format(stat.time_nonstop.elapsed(), 6,
								"%w") << "," << part_global.primalObjective
						<< "   " << part_global.dualObjective << "   "
						<< part_global.oneZeroAccuracy << "  gap "
						<< part_global.primalObjective
								- part_global.dualObjective <<

						endl;
				myfile << "intermediate_" << stat.last_obj_value << ","
						<< stat.generated_optimal_value << ","
						<< stat.last_obj_value - stat.generated_optimal_value
						<< "," << rr_iteration << ",IVAL" << ","
						<< boost::timer::format(stat.time_nonstop.elapsed(), 6,
								"%w") << "," << part_global.primalObjective
						<< "   " << part_global.dualObjective << "   "
						<< part_global.oneZeroAccuracy << "  gap "
						<< part_global.primalObjective
								- part_global.dualObjective <<

						endl;

				if (abs(stat.last_obj_value - stat.generated_optimal_value)
						> 0.00000000001) {
					continue_work = 1;
				}
//				printf("LAST Objective %f   %f\n", stat.last_obj_value, part.lambda);
			}
			boost::mpi::broadcast(world, continue_work, 0);
		}
		if (settings.show_prediction_accuracy) {
			PROF("prediction accuracy")
			stat.prediction_accuracy =
					compute_distributed_prediction_accuracy_for_svm_for_part(
							env, world, part_global);
		}
		if (settings.showIntermediateObjectiveValue && world.rank() == 0) {
			logStatisticsToFile(myfile, stat, settings, "detail_", true);
		}
		stat.time_other_overhed.stop();
		rr_iteration++;
	}
	stat.time_all_computation.stop();
}

#endif // ACDC_DISTRIBUTED_SOLVER
