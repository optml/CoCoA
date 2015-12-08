#ifndef ACDC_DISTRIBUTED_SYNCHRONOUS_SCHEMA
#define ACDC_DISTRIBUTED_SYNCHRONOUS_SCHEMA

#include "distributed_include.h"


#include "distributed_essentials.h"
#include "distributed_common.h"
#include "distributed_structures.h"
#include "data_distributor.h"

using namespace std;
namespace mpi = boost::mpi;

template<typename D, typename L>
void create_distribution_schema(mpi::communicator &world, ProblemData<L, D> &inst,
		data_distributor<L, D> &dataDistributor, std::list<L> & talk_to,
		std::list<std::list<L> > & what_to_exchange, std::list<L> & broadcasting,
		std::list<std::list<L> > & what_to_broadcast, DistributedSettings &settings) {

	typename std::list<L>::iterator it;

	int r = 0;
	mpi::request reqs[2 * world.size()];
	int tag = 0;

	if (world.rank() == 0) {

		double startfime = gettime_();

		std::vector < std::set<L> > row_indices;
		row_indices.resize(inst.m);

		/*
		 for (int part = 0; part < world.size(); part++) {
		 printf("Part:%d :", part);
		 for (int i = dataDistributor.countsPtr[part]; i < dataDistributor.countsPtr[part + 1]; i++) {
		 printf(" %d ", dataDistributor.indexes[i]);
		 }
		 printf("\n");
		 }
		 */

		// For each row, collect the parts it intersects
		for (int part = 0; part < world.size(); part++)
			// By doing: for all columns in that part
			for (int i = dataDistributor.countsPtr[part]; i < dataDistributor.countsPtr[part + 1]; i++) {
				int j = dataDistributor.indexes[i]; // j is the column-gid
				// go through the nnz in that column
				for (int nnz_i = inst.A_csc_col_ptr[j]; nnz_i < inst.A_csc_col_ptr[j + 1]; nnz_i++) {
					int row = inst.A_csc_row_idx[nnz_i];
					row_indices[row].insert(part);
				}
			}

		// For each part, look at what other parts it needs to talk to
		for (int part = 0; part < world.size(); part++) {

			std::set<L> from_part;
			std::list<L> send_to;
			std::list < std::list<L> > send_what;

			for (int row = 0; row < row_indices.size(); row++) {
				if (row_indices[row].find(part) != row_indices[row].end()) {
					std::set<L> temp = from_part;
					std::set_union(temp.begin(), temp.end(), row_indices[row].begin(), row_indices[row].end(),
							insert_iterator<std::set<L> > (from_part, from_part.begin()));
				}
			}
			from_part.erase(part);

			// NOTE: We cannot do "if (fromPart.size() == 0) continue;", because the couterparty
			// would be left waiting, hoping to receive the data, not knowing it should broadcast

			//
			if (from_part.size() >= settings.broadcast_treshold)
				broadcasting.push_back(part);
			else
				send_to.insert(send_to.begin(), from_part.begin(), from_part.end());

			/*
			 std::cout << "SENDTO: " << part << " ";
			 for (typename std::list<L>::iterator it = send_to.begin(); it != send_to.end(); it++) {
			 std::cout << *it << " ";
			 }
			 std::cout << std::endl;
			 */

			// NOTE: The exchange of data is symmetric
			if (part == 0)
				talk_to = send_to;
			else
				reqs[r++] = world.isend(part, 0, send_to);

			// NOTE: This need to work with empty bits in send_to
			if (settings.distributed == SynchronousSparse || settings.distributed == SynchronousSupersparse) {
				for (it = send_to.begin(); it != send_to.end(); it++) {
					std::list<L> rows_ids;
					for (int row = 0; row < row_indices.size(); row++) {
						if (row_indices[row].find(part) != row_indices[row].end() && row_indices[row].find(*it)
								!= row_indices[row].end()) {
							rows_ids.push_back(row);
						}
					}
					send_what.push_back(rows_ids);
				}
				if (part == 0)
					what_to_exchange = send_what;
				else
					reqs[r++] = world.isend(part, 1, send_what);
			}

		}

		// NOTE: We know (world.rank() == 0)
		broadcast(world, broadcasting, 0);

		if (settings.distributed == SynchronousSparse || settings.distributed == SynchronousSupersparse) {
			for (it = broadcasting.begin(); it != broadcasting.end(); it++) {
				std::list<L> rows_ids;
				for (int row = 0; row < row_indices.size(); row++) {
					if (row_indices[row].find(*it) != row_indices[row].end()) {
						rows_ids.push_back(row);
					}
				}
				what_to_broadcast.push_back(rows_ids);
			}
			broadcast(world, what_to_broadcast, 0);
		}

		mpi::wait_all(reqs, reqs + r);

		double endtime = gettime_();
		if (world.rank() == 0) // && settings.verbose)
			printf("Creating and distributing the distribution schema took %f\n", endtime - startfime);

	} else {
		world.recv(0, 0, talk_to);
		if (settings.distributed == SynchronousSparse || settings.distributed == SynchronousSupersparse)
			world.recv(0, 1, what_to_exchange);
		broadcast(world, broadcasting, 0);
		if (settings.distributed == SynchronousSparse || settings.distributed == SynchronousSupersparse) {
			std::list<L> dummy;
			what_to_broadcast.resize(broadcasting.size(), dummy);
			broadcast(world, what_to_broadcast, 0);
		}
	}

}

template<typename D, typename L>
void create_distribution_schema_for_multiple_sources(mpi::communicator &world, ProblemData<L, D> &inst,
		ProblemData<L, D> &part, data_distributor<L, D> &dataDistributor, DistributedSettings &settings,
		distributed_statistics &stat) {
	/*
	 * obtain bool of coordinate to broadcast
	 */
	std::vector<bool> to_broadcast(inst.m, false);


	L hypergraph_cut = 0;
	L total_rows = part.b.size();

	std::vector < std::set<L> > send_what(inst.m);

	stat.max_parts_per_row = 0;

	for (L row = 0; row < inst.m; row++) {
		std::set<L> tmp;
		for (L colId = inst.A_csr_row_ptr[row]; colId < inst.A_csr_row_ptr[row + 1]; colId++) {
			L col = inst.A_csr_col_idx[colId];
			tmp.insert(dataDistributor.columns_parts[col]);
		}
		if (tmp.size() > 0) {
			hypergraph_cut += tmp.size() - 1;
		}

		if (tmp.size() > stat.max_parts_per_row)
			stat.max_parts_per_row = tmp.size();
		if (tmp.size() < settings.broadcast_treshold) {
			send_what[row] = tmp;
		} else {
			to_broadcast[row] = true;
		}
	}
	stat.hypergraph_cut = hypergraph_cut;
	all_reduce(world, stat.hypergraph_cut, stat.hypergraph_cut, std::plus<long>());
	all_reduce(world, stat.max_parts_per_row, stat.max_parts_per_row, mpi::maximum<long>());

	std::vector < std::vector<bool> > all_to_broadcast(inst.m);
	gather(world, to_broadcast, all_to_broadcast, 0);
	if (world.rank() == 0) {
		to_broadcast.resize(part.m);
		int tmp_i = 0;
		for (L i = 0; i < world.size(); i++) {
			for (int j = 0; j < all_to_broadcast[i].size(); j++) {
				to_broadcast[tmp_i] = all_to_broadcast[i][j];
				tmp_i++;
			}
		}
	}
	broadcast(world, to_broadcast, 0);
	std::vector < std::vector<std::set<L> > > all_send_what;
	gather(world, send_what, all_send_what, 0);
	std::vector < std::set<L> > send_what_final(total_rows);
	if (world.rank() == 0) { //prepare final send_what data
		int tmp_i = 0;
		for (int part = 0; part < world.size(); part++) {
			for (int partial_m = 0; partial_m < all_send_what[part].size(); partial_m++) {
				if (!to_broadcast[tmp_i]) {
					for (typename std::set<L>::iterator i = all_send_what[part][partial_m].begin(); i
							!= all_send_what[part][partial_m].end(); i++) {
						send_what_final[dataDistributor.global_row_id_mapper[part] + partial_m].insert(*i);
					}
				}
				tmp_i++;
			}
		}
	}
	broadcast(world, send_what_final, 0);
	//everyone is going to produce his exchange queue
	const int my_rank = world.rank();
	const int w_size = world.size();

	dataDistributor.coordinates_of_updates.resize(w_size);
	dataDistributor.coordinates_of_reduce.resize(0);

	typename set<L>::iterator it;
	long total_numbers_which_has_to_be_exchanged = 0;
	int residual_rows_to_be_reduced = 0;

	for (L j = 0; j < send_what_final.size(); j++) {

		if (to_broadcast[j]) {
			dataDistributor.coordinates_of_reduce.push_back(j);
			residual_rows_to_be_reduced++;
		} else {
			it = send_what_final[j].find(my_rank);
			if (it != send_what_final[j].end()) {
				//add all nodes for this row
				for (typename std::set<L>::iterator i = send_what_final[j].begin(); i != send_what_final[j].end(); i++) {
					if (*i != my_rank) {
						dataDistributor.talking_members.insert(*i);
						total_numbers_which_has_to_be_exchanged++;
					}
					dataDistributor.coordinates_of_updates[*i].push_back(j);
				}
			}
		}
	}

	dataDistributor.residual_update_buffer_receive.resize(w_size);
	dataDistributor.residual_update_buffer_sent.resize(w_size);

	dataDistributor.residual_update_buffer_reduce_sent.resize(residual_rows_to_be_reduced, 0);
	dataDistributor.residual_update_buffer_reduce_receive.resize(residual_rows_to_be_reduced, 0);

	for (L partcnt = 0; partcnt < w_size; partcnt++) {
		dataDistributor.residual_update_buffer_sent[partcnt].resize(
				dataDistributor.coordinates_of_updates[partcnt].size());
		dataDistributor.residual_update_buffer_receive[partcnt].resize(
				dataDistributor.coordinates_of_updates[partcnt].size());
	}
	// Obtain some statistics about the dataset
	L max_nnz_per_column = 0;
	for (L col = 0; col < part.n; col++) {
		L tmp = part.A_csc_col_ptr[col + 1] - part.A_csc_col_ptr[col];
		if (tmp > max_nnz_per_column)
			max_nnz_per_column = tmp;
	}
	L max_nnz_per_row = 0;
	for (L row = 0; row < inst.m; row++) {
		L tmp = inst.A_csr_row_ptr[row + 1] - inst.A_csr_row_ptr[row];
		if (tmp > max_nnz_per_row)
			max_nnz_per_row = tmp;
	}

	L final_max_nnz_per_column = 0;
	L final_max_nnz_per_row = 0;
	reduce(world, max_nnz_per_column, final_max_nnz_per_column, mpi::maximum<L>(), 0);
	reduce(world, max_nnz_per_row, final_max_nnz_per_row, mpi::maximum<L>(), 0);

	//	printf("NODE%d, total number to be exchanged:%d\n", my_rank, total_numbers_which_has_to_be_exchanged);
	long final_total_numbers_which_has_to_be_exchanged = 0;
	reduce(world, total_numbers_which_has_to_be_exchanged, final_total_numbers_which_has_to_be_exchanged,
			std::plus<long>(), 0);

	stat.residual_rows_with_reduce = residual_rows_to_be_reduced;
	stat.max_nnz_per_row = final_max_nnz_per_row;
	stat.max_nnz_per_col = final_max_nnz_per_column;
	stat.floats_exchanged_per_comm_iter = final_total_numbers_which_has_to_be_exchanged;

	if (world.rank() == 0) {
		inst.sigma = 1.0 + (0.0 + (settings.totalThreads * world.size() * settings.iterationsPerThread)
				* (stat.max_nnz_per_row - 1)) / (inst.n + 0.0);
	}
	broadcast(world, inst.sigma, 0);
	part.sigma = inst.sigma;
}

template<typename D, typename L>
void create_distribution_schema_for_multiple_sources_old(mpi::communicator &world, ProblemData<L, D> &inst,
		ProblemData<L, D> &part, data_distributor<L, D> &dataDistributor, DistributedSettings &settings,
		distributed_statistics &stat) {
	L total_rows = part.b.size();
	std::vector < std::set<L> > send_what(total_rows);
	for (L row = 0; row < inst.m; row++) {
		for (L colId = inst.A_csr_row_ptr[row]; colId < inst.A_csr_row_ptr[row + 1]; colId++) {
			L col = inst.A_csr_col_idx[colId];
			send_what[row].insert(dataDistributor.columns_parts[col]);
		}
	}
	std::vector < std::vector<std::set<L> > > all_send_what;
	gather(world, send_what, all_send_what, 0);
	std::vector < std::set<L> > send_what_final(total_rows);
	std::vector < std::set<L> > reduce_what_final(total_rows);
	L hypergraph_cut = 0;
	if (world.rank() == 0) { //prepare final send_what data
		for (int part = 0; part < world.size(); part++) {
			for (int partial_m = 0; partial_m < all_send_what[part].size(); partial_m++) {
				if (all_send_what[part][partial_m].size() > 0)
					hypergraph_cut--;
				for (typename std::set<L>::iterator i = all_send_what[part][partial_m].begin(); i
						!= all_send_what[part][partial_m].end(); i++) {
					hypergraph_cut++;
					send_what_final[dataDistributor.global_row_id_mapper[part] + partial_m].insert(*i);
				}
			}
		}
		stat.hypergraph_cut = hypergraph_cut;
	}

	printf("STEP3 %d\n", world.rank());
	world.barrier();

	broadcast(world, send_what_final, 0);
	//everyone is going to produce his exchange queue
	const int my_rank = world.rank();
	const int w_size = world.size();
	dataDistributor.coordinates_of_updates.resize(w_size);
	dataDistributor.coordinates_of_reduce.resize(0);
	typename set<L>::iterator it;
	long total_numbers_which_has_to_be_exchanged = 0;
	int residual_rows_to_be_reduced = 0;
	stat.max_parts_per_row = 0;
	for (L j = 0; j < send_what_final.size(); j++) {
		if (send_what_final[j].size() > stat.max_parts_per_row)
			stat.max_parts_per_row = send_what_final[j].size();
		if (send_what_final[j].size() > settings.broadcast_treshold) {
			dataDistributor.coordinates_of_reduce.push_back(j);
			residual_rows_to_be_reduced++;
		} else {
			it = send_what_final[j].find(my_rank);
			if (it != send_what_final[j].end()) {
				//add all nodes for this row
				for (typename std::set<L>::iterator i = send_what_final[j].begin(); i != send_what_final[j].end(); i++) {
					if (*i != my_rank) {
						dataDistributor.talking_members.insert(*i);
						total_numbers_which_has_to_be_exchanged++;
					}
					dataDistributor.coordinates_of_updates[*i].push_back(j);
					//					printf("Node %d  compun with %d row %d\n", my_rank, *i, j);
				}
			}
		}
	}
	dataDistributor.residual_update_buffer_receive.resize(w_size);
	dataDistributor.residual_update_buffer_sent.resize(w_size);

	dataDistributor.residual_update_buffer_reduce_sent.resize(residual_rows_to_be_reduced, 0);
	dataDistributor.residual_update_buffer_reduce_receive.resize(residual_rows_to_be_reduced, 0);

	for (L partcnt = 0; partcnt < w_size; partcnt++) {
		dataDistributor.residual_update_buffer_sent[partcnt].resize(
				dataDistributor.coordinates_of_updates[partcnt].size());
		dataDistributor.residual_update_buffer_receive[partcnt].resize(
				dataDistributor.coordinates_of_updates[partcnt].size());
	}
	// Obtain some statistics about the dataset
	L max_nnz_per_column = 0;
	for (L col = 0; col < part.n; col++) {
		L tmp = part.A_csc_col_ptr[col + 1] - part.A_csc_col_ptr[col];
		if (tmp > max_nnz_per_column)
			max_nnz_per_column = tmp;
	}
	L max_nnz_per_row = 0;
	for (L row = 0; row < inst.m; row++) {
		L tmp = inst.A_csr_row_ptr[row + 1] - inst.A_csr_row_ptr[row];
		if (tmp > max_nnz_per_row)
			max_nnz_per_row = tmp;
	}

	L final_max_nnz_per_column = 0;
	L final_max_nnz_per_row = 0;
	reduce(world, max_nnz_per_column, final_max_nnz_per_column, mpi::maximum<L>(), 0);
	reduce(world, max_nnz_per_row, final_max_nnz_per_row, mpi::maximum<L>(), 0);

	//	printf("NODE%d, total number to be exchanged:%d\n", my_rank, total_numbers_which_has_to_be_exchanged);
	long final_total_numbers_which_has_to_be_exchanged = 0;
	reduce(world, total_numbers_which_has_to_be_exchanged, final_total_numbers_which_has_to_be_exchanged,
			std::plus<long>(), 0);

	stat.residual_rows_with_reduce = residual_rows_to_be_reduced;
	stat.max_nnz_per_row = final_max_nnz_per_row;
	stat.max_nnz_per_col = final_max_nnz_per_column;
	stat.floats_exchanged_per_comm_iter = final_total_numbers_which_has_to_be_exchanged;

	if (world.rank() == 0) {
		inst.sigma = 1.0 + (0.0 + (settings.totalThreads * world.size() * settings.iterationsPerThread)
				* (stat.max_nnz_per_row - 1)) / (inst.n + 0.0);
	}
	broadcast(world, inst.sigma, 0);
	part.sigma = inst.sigma;
}

#endif // ACDC_DISTRIBUTED_SYNCHRONOUS_SCHEMA
