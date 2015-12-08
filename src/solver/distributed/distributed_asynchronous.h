#ifndef ACDC_DISTRIBUTED_ASYNCHRONOUS
#define ACDC_DISTRIBUTED_ASYNCHRONOUS

#include "distributed_include.h"


#include "distributed_essentials.h"

//#include "solver/structures.h"
//#include "solver/l2l1/parallel/random_solver_nonoverlapping.h"

//#include "data_distributor.h"


//#include "distributed_common.h"

using namespace std;
namespace mpi = boost::mpi;

// This takes T + 1 rounds (!) of communication, but transfers only a modest amount of data
// TODO: Is there a lower bound for the communication complexity? This could be it.
//template<typename D, typename L>
//void shift_residuals(mpi::environment &env, mpi::communicator &world, std::vector<D> &residuals,
//		std::vector<D> &residual_updates, L round) {
//
//	// how many updates we want to receive in a single step
//	const int k = 1;
//	std::vector<D> empty_v(residuals.size(), 0);
//	std::vector < std::vector<D> > received(k, empty_v);
//
//	int r = 0;
//	const int msgs_each = 2;
//	mpi::request reqs[msgs_each + 1]; // NOTE: There is an extra request at the root node
//	int tag = (round * world.rank() * k * (msgs_each + 1)) % env.max_tag();
//
//	// get the thing going
//	if (world.rank() == 0) {
//		for (int i = 0; i < k; i++) {
//			int send_to = (world.rank() + 1 + i) % (world.size());
//			vsend(world, send_to, tag++, residual_updates);
//		}
//	}
//
//	// one pass
//	for (int i = 0; i < k; i++) {
//		int receive_from = (world.rank() - 1 - i + world.size()) % (world.size());
//		reqs[r++] = virecv(world, receive_from, tag++, received[i]);
//	}
//	mpi::wait_all(reqs, reqs + r);
//	// the actual update
//	for (int i = 0; i < k; i++)
//		for (int j = 0; j < residuals.size(); j++) {
//			residuals[j] += received[i][j];
//			residual_updates[j] += received[i][j];
//		}
//
//	for (int i = 0; i < k; i++) {
//		int send_to = (world.rank() + 1 + i) % (world.size());
//		reqs[r++] = visend(world, send_to, tag++, residual_updates);
//	}
//
//}

// This takes 1 round of communication, but performs a lot of computations
//template<typename D, typename L>
//void shift_residuals_buffered(mpi::environment &env, mpi::communicator &world,
//		std::list<std::vector<D> > &buffer, std::vector<D> &residuals, std::vector<D> &residual_updates,
//		DistributedSettings &settings, L round) {
//
//	// NOTE: THIS IS THE MEAT
//	buffer.pop_back();
//	buffer.push_front(residual_updates);
//
//	const int msgs_each = 2;
//	int tag = round % env.max_tag();
//
//	int r = 0;
//	mpi::request reqs[msgs_each];
//
//	int send_to = (world.rank() + 1) % (world.size());
//	// NOTE: Best not replaced by visend
//	reqs[r++] = world.isend(send_to, tag, buffer);
//
//	std::vector<D> temp(residuals.size(), 0);
//	int receive_from = (world.rank() - 1 + world.size()) % (world.size());
//	reqs[r++] = world.irecv(receive_from, tag, buffer);
//
//	mpi::wait_all(reqs, reqs + r);
//
//	for (int i = 0; i < residuals.size(); i++) {
//		for (typename std::list<std::vector<D> >::iterator it = buffer.begin(); it != buffer.end(); it++)
//			residuals[i] += it->at(i);
//		residuals[i] += residual_updates[i];
//	}
//
//}

// This takes 1 round of communication, transfers less
//template<typename D, typename L>
//void shift_residuals_buffered_streamlined(mpi::environment &env, mpi::communicator &world,
//		std::list<std::vector<D> > &buffer, std::vector<D> &past_update, std::vector<D> &residuals,
//		std::vector<D> &residual_updates, DistributedSettings &settings, L round) {
//
//	int tag = round % env.max_tag();
//
//	int r = 0;
//	const int msgs_each = 2;
//	mpi::request reqs[msgs_each];
//	std::vector<D> received(residuals.size(), 0);
//	int receive_from = (world.rank() - 1 + world.size()) % (world.size());
//	reqs[r++] = virecv(world, receive_from, tag, received);
//
//	cblas_sum_of_vectors(past_update, residual_updates);
//	int send_to = (world.rank() + 1) % (world.size());
//	reqs[r++] = visend(world, send_to, tag, past_update);
//
//	mpi::wait_all(reqs, reqs + r);
//	for (int i = 0; i < residuals.size(); i++) {
//		received[i] -= buffer.back()[i];
//		residuals[i] += received[i] + residual_updates[i];
//	}
//	buffer.pop_back();
//	buffer.push_front(residual_updates);
//	past_update = received;
//
//}

// This takes 1 round of communication, transfers less
//template<typename D, typename L>
//void shift_residuals_buffered_streamlined_v2(mpi::environment &env, mpi::communicator &world,
//		std::list<std::vector<D> > &buffer, std::vector<D> &past_update, std::vector<D> &residuals,
//		std::vector<D> &residual_updates, DistributedSettings &settings, L round) {
//
//	int tag = round % env.max_tag();
//
//	int r = 0;
//	const int msgs_each = 2;
//	mpi::request reqs[msgs_each];
//	std::vector<D> received(residuals.size(), 0);
//	int receive_from = (world.rank() - 1 + world.size()) % (world.size());
//	reqs[r++] = virecv(world, receive_from, tag, received);
//
//	cblas_sum_of_vectors(past_update, residual_updates);
//	int send_to = (world.rank() + 1) % (world.size());
//	reqs[r++] = visend(world, send_to, tag, past_update);
//
//	mpi::wait_all(reqs, reqs + r);
//	for (int i = 0; i < residuals.size(); i++) {
//		received[i] -= buffer.back()[i];
//		residuals[i] += received[i] + residual_updates[i];
//	}
//	buffer.pop_back();
//	buffer.push_front(residual_updates);
//	past_update = received;
//
//}

// This takes 1 round of communication, transfers less,
// has all the arrays pre-allocated
template<typename D, typename L>
void shift_residuals_buffered_streamlined_optimized(
		mpi::communicator &world,
		std::vector<D> &residual_updates_to_exchange,
		std::vector<D> &residuals,
		std::vector<D> &my_past_residual_updates,
		std::vector<D> &my_exchange_storage,
		L &exchange_data,
		L &AsynchronousStreamlinedOptimizedUpdate
		//,
		//DistributedSettings &settings, L round
		) {
	int tag = 0;//round % env.max_tag();
	int r = 0;
	const int msgs_each = 2;
	mpi::request reqs[msgs_each];
	int tmpI = residuals.size() * exchange_data;
	int receive_from = (world.rank() - 1 + world.size()) % (world.size());
	reqs[r++] = virecv(world, receive_from, tag, &my_exchange_storage[tmpI], residuals.size());

	L to_sent_shift = ((exchange_data + 1) % 2) * residuals.size();
	L current_update = AsynchronousStreamlinedOptimizedUpdate * residuals.size();

	for (L i=0;i<residuals.size();i++){
	my_past_residual_updates[current_update+i]=	residual_updates_to_exchange[i];
	}

	cblas_sum_of_vectors(residuals.size(), &my_exchange_storage[to_sent_shift],
			&my_past_residual_updates[current_update]);

	int send_to = (world.rank() + 1) % (world.size());
	reqs[r++] = visend(world, send_to, tag, &my_exchange_storage[to_sent_shift], residuals.size());

	cblas_sum_of_vectors(residuals.size(), &residuals[0], &my_past_residual_updates[current_update]);

	mpi::wait_all(reqs, reqs + r);
	int tmpI2 = ((AsynchronousStreamlinedOptimizedUpdate + 1) % world.size()) * residuals.size();

	cblas_subtract_of_vectors(residuals.size(), &my_exchange_storage[tmpI], &my_past_residual_updates[tmpI2]);

	cblas_sum_of_vectors(residuals.size(), &residuals[0], &my_exchange_storage[tmpI]);

	exchange_data = ((exchange_data + 1) % 2);
	AsynchronousStreamlinedOptimizedUpdate = (AsynchronousStreamlinedOptimizedUpdate + 1) % (world.size());
}

// This is a plain generalisation of the token ring setup
// This takes 1 round of communication, transfers more than streamlined, but the synchronisation time is less
// This is a generalisation of the asynchronous (token) ring setup
// This takes 1 round of communication, transfers more than streamlined,
// but the synchronisation time is less
//template<typename D, typename L>
//void shift_residuals_torus(mpi::environment &env, mpi::communicator &world,
//		std::vector<D> &residual_updates_tosum, std::vector<D> &exchanged, std::list<std::vector<D> > &buffer,
//		std::vector<D> &past_update, std::vector<D> &residuals, std::vector<D> &residual_updates,
//		DistributedSettings &settings, L round) {
//
//	int k = settings.torus_width;
//
//	// NOTE: on 4 computers, width 2 does not work
//	if (settings.topology.count_rungs(world.size(), k) < 3) {
//		std::cerr << "ERROR: Too few computers to run Torus of this width. Need 3+ rungs." << std::endl;
//		return;
//	}
//
//	int rung_root = settings.topology.this_rung_index(world.size(), world.rank(), k, 0);
//	int rung_root_next = settings.topology.next_rung_index(world.size(), world.rank(), k, 0);
//	int rung_root_previous = settings.topology.previous_rung_index(world.size(), world.rank(), k, 0);
//
//	// FIXME DELETE
//	//	printf("%d    %d<%d<%d \n",world.rank(),rung_root_previous,rung_root,rung_root_next);
//	//	world.barrier();
//	//	printf("%d    %d<%d<%d \n",world.rank(),rung_root_previous,rung_root,rung_root_next);
//	//	world.barrier();
//
//	int tag = round % env.max_tag();
//	int r = 0;
//	const int msgs_each = 2 + 2 * k;
//	std::vector<mpi::request> reqs(msgs_each);
//
//	if (world.rank() != rung_root) {
//
//		// outside of the root node, we just share our update
//		reqs[r++] = visend(world, rung_root, tag, residual_updates);
//
//		// wait for the update from this rung's root
//		reqs[r++] = virecv(world, rung_root, tag, exchanged);
//
//		mpi::wait_all(&reqs[0], &reqs[r]);
//
//		// and apply the update
//		for (int i = 0; i < residuals.size(); i++) {
//			residuals[i] += exchanged[i];
//		}
//
//	} else {
//
//		reqs[r++] = visend(world, rung_root_next, tag, past_update);
//		reqs[r++] = virecv(world, rung_root_previous, tag, exchanged);
//
//		int i, j;
//		for (i = 1; i < k; i++)
//			reqs[r++] = virecv(world, settings.topology.this_rung_index(world.size(), world.rank(), k, i), tag,
//					&residual_updates_tosum[(i - 1) * residuals.size()], residuals.size());
//		mpi::wait_all(&reqs[0], &reqs[r]);
//		int rr = r;
//		for (j = 1; j < k; j++) {
//			for (i = 0; i < residuals.size(); i++) {
//				residual_updates[i] += residual_updates_tosum[(j - 1) * residuals.size() + i];
//			}
//		}
//
//		for (i = 0; i < residuals.size(); i++) {
//			exchanged[i] += residual_updates[i] - buffer.back()[i];
//			residuals[i] += exchanged[i];
//		}
//
//		for (i = 1; i < k; i++) {
//			reqs[rr++] = visend(world, settings.topology.this_rung_index(world.size(), world.rank(), k, i), tag,
//					exchanged);
//		}
//
//		buffer.pop_back();
//		buffer.push_front(residual_updates);
//
//		past_update = exchanged;
//		mpi::wait_all(&reqs[r], &reqs[rr]);
//	}
//
//}

// This is a plain generalisation of the token ring setup
// This takes 1 round of communication, transfers more than streamlined, but the synchronisation time is less
// This is a generalisation of the asynchronous (token) ring setup
// This takes 1 round of communication, transfers more than streamlined,
// but the synchronisation time is less
//template<typename D, typename L>
//void shift_residuals_torus_opt(mpi::environment &env, mpi::communicator &world,
//		std::vector<D> &residual_updates_tosum, std::vector<D> &exchanged, std::list<std::vector<D> > &buffer,
//		std::vector<D> &past_update, std::vector<D> &residuals, std::vector<D> &residual_updates,
//		DistributedSettings &settings, L round, mpi::communicator &local_rung_communicator) {
//
//	int k = settings.torus_width;
//
//	// NOTE: on 4 computers, width 2 does not work
//	if (settings.topology.count_rungs(world.size(), k) < 3) {
//		std::cerr << "ERROR: Too few computers to run Torus of this width. Need 3+ rungs." << std::endl;
//		return;
//	}
//	int rung_root = settings.topology.this_rung_index(world.size(), world.rank(), k, 0);
//	int rung_root_next = settings.topology.next_rung_index(world.size(), world.rank(), k, 0);
//	int rung_root_previous = settings.topology.previous_rung_index(world.size(), world.rank(), k, 0);
//
//	int tag = round % env.max_tag();
//	int r = 0;
//	int i;int j;
//	const int msgs_each = 2 + 2 * k;
//	std::vector<mpi::request> reqs(msgs_each);
//
//	if (world.rank() == rung_root) {
//		reqs[r++] = visend(world, rung_root_next, tag, past_update);
//		reqs[r++] = virecv(world, rung_root_previous, tag, exchanged);
//	}
//	// outside of the root node, we just share our update
//	//vreduce(local_rung_communicator, residual_updates, residual_updates_tosum, 0);
//
//	if (world.rank()!=rung_root){
//			reqs[r++] = visend(world, rung_root, tag, residual_updates);
//			reqs[r++] = virecv(world, rung_root, tag, residuals);
//			mpi::wait_all(&reqs[0], &reqs[r]);
//	}
//
//	if (world.rank() == rung_root) {
//		for (i = 1; i < k; i++)
//				reqs[r++] = virecv(world, settings.topology.this_rung_index(world.size(), world.rank(), k, i), tag,
//						&residual_updates_tosum[(i - 1) * residuals.size()], residuals.size());
//		mpi::wait_all(&reqs[0], &reqs[r]);
//int rr=r;
//		for (j = 1; j < k; j++) {
//			cblas_sum_of_vectors(residuals.size(), &residual_updates[0],
//					&residual_updates_tosum[(j - 1) * residuals.size()]);
//				}
//
//		cblas_sum_of_vectors(residuals.size(), &exchanged[0], &residual_updates[0]);
//		//cblas_sum_of_vectors(residuals.size(), &exchanged[0], &residual_updates_tosum[0]);
//		cblas_subtract_of_vectors(residuals.size(), &exchanged[0], &buffer.back()[0]);
//		cblas_sum_of_vectors(residuals.size(), &residuals[0], &exchanged[0]);
//
//		for (i = 1; i < k; i++) {
//					reqs[rr++] = visend(world, settings.topology.this_rung_index(world.size(), world.rank(), k, i), tag,
//							residuals);
//				}
//
//		buffer.pop_back();
////		buffer.push_front(residual_updates_tosum);
//				buffer.push_front(residual_updates);
//		past_update = exchanged;
//		mpi::wait_all(&reqs[r], &reqs [rr]);
//	}
//	vbroadcast(local_rung_communicator, residuals, 0);
//
//}

//template<typename D, typename L>
//void shift_residuals_torus_opt_collectives(mpi::environment &env, mpi::communicator &world,
//		std::vector<D> &residual_updates_tosum, std::vector<D> &exchanged, std::list<std::vector<D> > &buffer,
//		std::vector<D> &past_update, std::vector<D> &residuals, std::vector<D> &residual_updates,
//		DistributedSettings &settings, L round, mpi::communicator &local_rung_communicator) {
//
//	int k = settings.torus_width;
//
//	// NOTE: on 4 computers, width 2 does not work
//	if (settings.topology.count_rungs(world.size(), k) < 3) {
//		std::cerr << "ERROR: Too few computers to run Torus of this width. Need 3+ rungs." << std::endl;
//		return;
//	}
//	int rung_root = settings.topology.this_rung_index(world.size(), world.rank(), k, 0);
//	int rung_root_next = settings.topology.next_rung_index(world.size(), world.rank(), k, 0);
//	int rung_root_previous = settings.topology.previous_rung_index(world.size(), world.rank(), k, 0);
//
//	int tag = round % env.max_tag();
//	int r = 0;
//	int i;int j;
//	const int msgs_each = 2 + 2 * k;
//	std::vector<mpi::request> reqs(msgs_each);
//
//	if (world.rank() == rung_root) {
//		reqs[r++] = visend(world, rung_root_next, tag, past_update);
//		reqs[r++] = virecv(world, rung_root_previous, tag, exchanged);
//	}
//	// outside of the root node, we just share our update
//	vreduce(local_rung_communicator, residual_updates, residual_updates_tosum, 0);
//
////	if (world.rank()!=rung_root){
////			reqs[r++] = visend(world, rung_root, tag, residual_updates);
////			reqs[r++] = virecv(world, rung_root, tag, residuals);
////			mpi::wait_all(&reqs[0], &reqs[r]);
////	}
//
//	if (world.rank() == rung_root) {
////		for (i = 1; i < k; i++)
////				reqs[r++] = virecv(world, settings.topology.this_rung_index(world.size(), world.rank(), k, i), tag,
////						&residual_updates_tosum[(i - 1) * residuals.size()], residuals.size());
////		mpi::wait_all(&reqs[0], &reqs[r]);
//int rr=r;
//		for (j = 1; j < k; j++) {
//			cblas_sum_of_vectors(residuals.size(), &residual_updates[0],
//					&residual_updates_tosum[(j - 1) * residuals.size()]);
//				}
//
//		cblas_sum_of_vectors(residuals.size(), &exchanged[0], &residual_updates[0]);
//		//cblas_sum_of_vectors(residuals.size(), &exchanged[0], &residual_updates_tosum[0]);
//		cblas_subtract_of_vectors(residuals.size(), &exchanged[0], &buffer.back()[0]);
//		cblas_sum_of_vectors(residuals.size(), &residuals[0], &exchanged[0]);
//
////		for (i = 1; i < k; i++) {
////					reqs[rr++] = visend(world, settings.topology.this_rung_index(world.size(), world.rank(), k, i), tag,
////							residuals);
////				}
//
//		buffer.pop_back();
////		buffer.push_front(residual_updates_tosum);
//				buffer.push_front(residual_updates);
//		past_update = exchanged;
////		mpi::wait_all(&reqs[r], &reqs [rr]);
//	}
//	vbroadcast(local_rung_communicator, residuals, 0);
//
//}



// see shift_residuals_torus_collectives below for the use
//void torus_collectives_prepare(mpi::environment &env, mpi::communicator &world, mpi::communicator &rung,
//		DistributedSettings &settings) {
//
//	int k = settings.torus_width;
//	int r = settings.topology.this_rung(world.size(), world.rank(), k);
//	rung = world.split(r);
//}

// This replaces the point-to-point communication along
// the torus with collective (reduce, broadcast)
//template<typename D, typename L>
//void shift_residuals_torus_collectives(mpi::environment &env, mpi::communicator &world,
//		mpi::communicator &rung, std::list<std::vector<D> > &buffer, std::vector<D> &past_update,
//		std::vector<D> &residuals, std::vector<D> &residual_updates, DistributedSettings &settings, L round) {
//
//	int k = settings.torus_width;
//
//	int rung_no = settings.topology.this_rung(world.size(), world.rank(), k);
//	int rung_no_previous = settings.topology.previous_rung(world.size(), world.rank(), k);
//	int rung_root = settings.topology.this_rung_index(world.size(), world.rank(), k, 0);
//	int rung_root_next = settings.topology.next_rung_index(world.size(), world.rank(), k, 0);
//	int rung_root_previous = settings.topology.previous_rung_index(world.size(), world.rank(), k, 0);
//
//	std::vector<D> residual_updates_summed(residuals.size(), 0);
//	std::vector<D> exchanged(residuals.size(), 0);
//
//	if (world.rank() != rung_root) {
//
//		// outside of the root node, we just provide our update to others
//		vreduce(rung, residual_updates, residual_updates_summed, 0);
//		// wait for the update from this rung's root
//		vbroadcast(rung, exchanged, 0);
//
//		// and apply the update
//		cblas_sum_of_vectors(residuals, exchanged);
//
//	} else {
//
//		int tag = round % env.max_tag();
//		int r = 0;
//		const int msgs_each = 2;
//		mpi::request reqs[msgs_each];
//
//		reqs[r++] = visend(world, rung_root_next, tag, past_update);
//		reqs[r++] = virecv(world, rung_root_previous, tag, exchanged);
//
//		if (k > 1)
//			vreduce(rung, residual_updates, residual_updates_summed, 0);
//
//		mpi::wait_all(reqs, reqs + r);
//
//		for (int i = 0; i < residuals.size(); i++) {
//			exchanged[i] += residual_updates_summed[i] - buffer.back()[i];
//			residuals[i] += exchanged[i];
//		}
//
//		vbroadcast(rung, exchanged, 0);
//
//		buffer.pop_back();
//		buffer.push_front(residual_updates_summed);
//
//		past_update = exchanged;
//	}
//
//}

#endif // ASYNCHRONOUS
